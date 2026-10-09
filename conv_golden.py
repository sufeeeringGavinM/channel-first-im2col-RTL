"""Lab 01: convolution as a loop nest.

Week 2 milestone M0. This is the golden model for the convolution you build in
SystemVerilog later in the course.

Conventions, fixed for the whole course:
    Input layout   NCHW, shape (N, C, H, W)
    Weight layout  KCRS, shape (K, C, R, S)
    Output layout  NKPQ, shape (N, K, P, Q)
    Padding        zero, symmetric, (pad_i, pad_j)
    Stride         (stride_i, stride_j)
    Dilation       (dilation_i, dilation_j)
    Bias           none in this lab
    Data type      Python ints only, never floats

Tensors are flat lists plus a shape tuple, never nested lists. Hardware has a
flat address space, so the index arithmetic is part of the exercise.

Only the `random` module may be imported. No other modules can be imported.

Fill in every function marked TODO. Do not change any function signature.
"""

import random

# ---------------------------------------------------------------------------
# Given helpers
# ---------------------------------------------------------------------------


def num_elements(shape):
    """Return the number of elements in a shape.

    Args:
        shape: tuple of dimension sizes.

    Returns:
        Product of the dimensions.
    """
    n = 1
    for d in shape:
        n *= d
    return n


def random_tensor(shape, lo=-128, hi=127, seed=42):
    """Build a flat tensor of random integers.

    Args:
        shape: tuple of dimension sizes.
        lo: smallest value, inclusive.
        hi: largest value, inclusive.
        seed: optional seed for a reproducible result.

    Returns:
        Flat list of `num_elements(shape)` integers.
    """
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(num_elements(shape))]


# ---------------------------------------------------------------------------
# Part A: convolution arithmetic
# ---------------------------------------------------------------------------


def conv_output_size(in_size, kernel, pad, stride):
    """Compute the output size along one spatial axis.

    Derive the formula from the definition of the sliding window rather than
    looking it up.

    Args:
        in_size: input size, H or W.
        kernel: kernel size, R or S.
        pad: zero padding added to each side of the axis.
        stride: step between window positions.

    Returns:
        Output size, P or Q.

    Raises:
        ValueError: if stride < 1, pad < 0, in_size < 1, kernel < 1, or the
            kernel does not fit inside the padded input.
    """

    if in_size < 1:
        raise ValueError("Input size below 1")
    if kernel < 1:
        raise ValueError("Filter size below 1")
    if stride < 1:
        raise ValueError("Stride below 1")
    if pad < 0:
        raise ValueError("Padding is negative")
    if kernel > (in_size + (2 * pad)):
        raise ValueError("Given filter size can't fit inside padded input size")

    # 1. Input size with padding on both sides = in_size + 2*pad;
    # 2. Subtract with what the filter takes up as it goes across the input: -kernel
      # Imagine you have padded input with indices [0 1 2 3 4 5 6] and kernel size 3
      # With center of kernel at i=0, the kernel "falls off". The same happens at i=6
      # The only valid kernel positions after the starting position i=1 is i=2,3,4,5.
      # This is 7-3 = 4 positions or padded input size minus kernel size.
    # 3. Then divide by the jump step of the kernel or the stride
    # 4. Then a plus one outside the expression above to account for the starting position of the kernel
      # More on this (why not add one on the numerator of the expression with the stride?)
        # Imagine padded input [0 1 2 3 4 5 6 7 8 9] with stride 2, kernel size 3
          # i=1, 3, 5, 7 are the valid kernel positions and thus output size is 4.
        # With formula (in-size + 2*pad - kernel)/stride, we get (10-3)//2 = 7//2 = 3 (it rounds down from 3.5).
        # We have to add one after the expression!

    # All in all, we can imagine the expression with the stride at the denominator
    # to represent the jumps after the initial kernel position,
    # and the +1 as the initial position itself.

    return ((in_size + (2 * pad) - kernel) // stride) + 1


def flat_index(shape, i0, i1, i2, i3):
    """Map a 4-D coordinate to an offset in a row-major flat list.

    The last dimension is contiguous.

    Args:
        shape: 4-tuple of dimension sizes.
        i0: index along the first dimension, N or K.
        i1: index along the second dimension, C.
        i2: index along the third dimension, H, R or P.
        i3: index along the fourth dimension, W, S or Q.

    Returns:
        Flat offset.

    Raises:
        ValueError: if any index is outside its dimension.
    """

    I0, I1, I2, I3 = shape
    coords = (i0, i1, i2, i3)
    dims = (I0, I1, I2, I3)

    # Check each coord
    for coord, dim in zip(coords, dims):
        if not (0 <= coord < dim):  # Dim is a count and coord is zero-indexed!
            raise ValueError("Coordinate found outside corresponding dimension")

    # We are looking for all that came before the coordinate.
    # Say we have an input which has dimensions NCHW...
    # Our coordinate is in the wth column of the hth row of the cth channel of
    # the nth image; N images, C channels, H rows, W columns
      # nth image: the n images before the coordinate is n*C*H*W
      # cth channel: the c channels before the coordinate is c*H*W
      # hth row: the h rows before the coordinate is h*W
      # wth column: the w columns before the coordinate is w

    # In the example above, (i0, i1, i2, i3) = (n,c,h,w) and (I0,I1,I2,I3) = (N,C,H,W)

    # The above is generalizable to the weight and output given the order of their shape is correct

    return (i0 * I1 * I2 * I3) + (i1 * I2 * I3) + (i2 * I3) + i3

    # This can be factored such that there are three multiplies but we can see
    # the pattern easier this way

# ---------------------------------------------------------------------------
# Part B: the loop nest
# ---------------------------------------------------------------------------


def conv2d(ifm, ifm_shape, wgt, wgt_shape, stride=(1, 1), pad=(0, 0), dilation=(1, 1)):
    """Run a 2-D convolution as a seven-deep loop nest.

    Handle padding by testing whether the input coordinate
     falls inside the feature map, not by building a padded copy.

     Args:
         ifm: flat list of input activations, NCHW order.
         ifm_shape: 4-tuple (N, C, H, W).
         wgt: flat list of weights, KCRS order.
         wgt_shape: 4-tuple (K, C, R, S).
         stride: (stride_i, stride_j).
         pad: (pad_i, pad_j), applied to both sides of each axis.
         dilation: (dilation_i, dilation_j), positive integer tap spacing.
             Defaults to (1, 1), preserving existing calls.

     Returns:
         Tuple of (flat output list in NKPQ order, output shape (N, K, P, Q)).

     Raises:
         ValueError: if the shapes are not rank 4, the channel counts disagree,
             a data list length does not match its shape, or dilation is invalid.
             Spatial sizes, padding, stride, and dilated kernel fit are validated
             by conv_output_size.
    """
    # Require rank-four shapes
    if len(ifm_shape) != 4:
        raise ValueError("ifm_shape isn't rank 4")
    if len(wgt_shape) != 4:
        raise ValueError("wgt_shape isn't rank 4")

    N, C_i, H, W = ifm_shape
    K, C_w, R, S = wgt_shape

    # Require IFM and WGT channel dimensions to matchs
    if C_i != C_w:
        raise ValueError("IFM and WGT channel dimensions don't match")

    # Require flat-list length to match shape
    if len(ifm) != N * C_i * H * W:
        raise ValueError("IFM flat-list length doesn't match ifm_shape")

    if len(wgt) != K * C_w * R * S:
        raise ValueError("WGT flat-list length doesn't match wgt_shape")

    # Obtain P and Q using conv_output_size

    stride_i, stride_j = stride
    pad_i, pad_j = pad
    dilation_i, dilation_j = dilation
    if any(not isinstance(d, int) or d < 1 for d in (dilation_i, dilation_j)):
        raise ValueError("Dilation must contain positive integers")
    if R < 1 or S < 1:
        raise ValueError("Filter size below 1")

    extent_i = dilation_i * (R - 1) + 1
    extent_j = dilation_j * (S - 1) + 1
    P = conv_output_size(H, extent_i, pad_i, stride_i)
    Q = conv_output_size(W, extent_j, pad_j, stride_j)

    ofm_shape = (N, K, P, Q)
    ofm = [0] * (N * K * P * Q)

    for n in range(N): # nth output image OR nth input image
        for k in range(K): # nth output image's kth output channel OR kth set of filters
            for p in range(P): # nth output image's kth output channel's pth row
                for q in range(Q): # nth output image's kth output channel's pth row's qth column
                    accumulated = 0
                    out_off = flat_index(ofm_shape, n, k, p, q)
                    # accumulates a spot in C_i channels of the nth image to one output pixel at (n,k,p,q)
                    for c in range(C_i):
                        for r in range(R):
                            for s in range(S):
                                h = p * stride_i - pad_i + r * dilation_i
                                w = q * stride_j - pad_j + s * dilation_j
                                if (0 <= h < H) and (0 <= w < W):
                                    ifm_off = flat_index(ifm_shape, n, c, h, w)
                                    wgt_off = flat_index(wgt_shape, k, c, r, s)
                                    accumulated += ifm[ifm_off] * wgt[wgt_off]
                    ofm[out_off] = accumulated

    return ofm, ofm_shape


# ---------------------------------------------------------------------------
# Part C: your tests
#
# Write these yourself. Each one must fail loudly, with an AssertionError that
# says what was wrong, when conv2d is wrong.
# ---------------------------------------------------------------------------


def test_output_size():
    """Check conv_output_size against hand-derived cases and bad arguments.

    Cover at least: no padding, padding that keeps the size the same, a stride
    above 1, a kernel as large as the input, and arguments that must raise an error.

    Raises:
        AssertionError: if any case is wrong.
    """

    #def conv_output_size(in_size, kernel, pad, stride):

    # 1. no padding

    ans = conv_output_size(10,3,0,1)

    # (1,2,3,4,5,6,7,8,9,10)
    # valid positions = 2,3,4,5,6,7,8,9 = 8 output values

    assert ans == 8, f'no padding case: expected 8, got {ans}'

    # 2. padding that keeps the (output) size the same (as the input size)

    ans = conv_output_size(10,3,1,1)

    # (x,1,2,3,4,5,6,7,8,9,10,x)
    # valid positions = 1 to 10 = 10 output values (same as input size)

    assert ans == 10, f'same size case: expected 10, got {ans}'

    # 3. stride above 1

    ans = conv_output_size(10,2,0,2)

    # (1,2,3,4,5,6,7,8,9,10)
    # valid windows: (1,2), (3,4), (5,6), (7,8), (9,10) = 5 output values

    assert ans == 5, f'stride above 1 case: expected 5, got {ans}'

    # 4. kernel as large as input

    ans = conv_output_size(1000,1000,0,1)

    assert ans == 1, f'kernel same size as input case: expected 1, got {ans}'

    # bad inputs:

    # 1. stride < 1

    try: conv_output_size(10,10,0,0)
    except ValueError: pass
    else: raise AssertionError("expected ValueError from stride < 1")


    # 2. pad < 0

    try: conv_output_size(10,10,-1,1)
    except ValueError: pass
    else: raise AssertionError("expected ValueError from pad < 0")

    # 3. in_size < 1

    try: conv_output_size(0,1,4,1) # pad to make sure it's due to in_size and not kernel not fitting padded input
    except ValueError: pass
    else: raise AssertionError("expected ValueError from in_size < 1")

    # 4. kernel < 1

    try: conv_output_size(10,0,0,1)
    except ValueError: pass
    else: raise AssertionError("expected ValueError from kernel < 1")

    # 5. kernel doesn't fit padded input

    try: conv_output_size(10,13,1,0)
    except ValueError: pass
    else: raise AssertionError("expected ValueError from kernel not fitting padded input")


def test_known_case():
    """Check one convolution you computed by hand.

    Use the worked example in the handout, then add a second case of your own
    with more than one input channel and more than one output channel.

    Raises:
        AssertionError: if the output differs from the hand computation.
    """
    ifm = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    wgt = [1, 2, 3, 4]
    got, shape = conv2d(ifm, (1, 1, 3, 3), wgt, (1, 1, 2, 2))
    assert shape == (1, 1, 2, 2), shape
    assert got == [37, 47, 67, 77], got

    # More Test Case!

    ifm = [1,2, # first image, first channel
           3,4,

           5,6, # first image, second channel
           7,8,

           9,10, # second image, first channel
           11,12,

           13,14, # second image, second channel
           15,16]

    wgt = [5,6, # first kernel, first channel
           7,8,

           1,2, # first kernel, second channel
           3,4,

           13,14, # second kernel, first channel
           15,16,

           9,10, # second kernel, second channel
           11,12]

    ifm_shape = (2,2,2,2) # Two images, two channels each, 2-tall height, 2-wide width
    wgt_shape = (2,2,2,2) # Two kernels, two channels each, 2-tall height, 2-wide width

    got, shape = conv2d(ifm, ifm_shape, wgt, wgt_shape)

    # Let's first walk through what the hell the output of this is supposed to look like before the assertion:

      # Start with the first image
          # The first output channel, which is the first image against the first kernel:
    '''
            *Due to the dimensions, the convolution is essentially just a dot product for this test case.
            *The 1D arrays here are really 2D, but flattened down for ease of viewing!

            First input channel convolved by first kernel channel:
            [1,2,3,4] * [5,6,7,8] = [1*5 + 2*6 + 3*7 + 4*8] = [70]
            Then the second input channel convolved with the second kernel channel
            [5,6,7,8] * [1,2,3,4] = [70]

            Then, sum these two to get the first output channel:
            [70] + [70] = [140] (yes, the output channels are just 1x1 matrices)
    '''
          # The second output channel, which is the first image against the second kernel:
    '''
            [1,2,3,4] * [13,14,15,16] = [13+28+45+64] = 150
            [5,6,7,8] * [9,10,11,12] = [45+60+77+96] = 278

            [150+278] = [428]
    '''
        # Then do the same to the second image against the same kernels to get the second output image's channels.
    '''

            first output channel:
            [9,10,11,12] * [5,6,7,8] = 278
            [13,14,15,16] * [1,2,3,4] = 150

            [428] = first output channel of second output image

            second output channel:
            [9,10,11,12] * [13,14,15,16] = [117 + 140 + 165 + 192] = 614
            [13,14,15,16] * [9,10,11,12] = 614

            [1228] = second output channel of second output image
    '''

        # So, putting all of that together,
    ''' first output image:
                first output channel: 140
                second output channel: 428
              second output image:
                first output channel: 428
                second output channel: 1228

              final flat list: [140,428,428,1228]
              final shape: [2,2,1,1]

    '''

    assert got == [140,428,428,1228], f'Got {got}, expected {[140,428,428,1228]}'
    assert shape == (2,2,1,1)




def test_shift_positions():
    """Check that a single non-zero weight copies a shifted input.

    With a 3x3 kernel that is zero everywhere except at tap (r0, s0), and
    pad=(1, 1), stride=(1, 1), the output is the input shifted by (r0-1, s0-1)
    with zeros shifted in at the border. Check all nine tap positions.

    This is the test that catches a flipped kernel and a padding sign error.

    Raises:
        AssertionError: if any tap position is wrong.
    """

    # Example:

    # [1,2,3,4
    #  5,6,7,8
    #  9,10,11,12,
    #  13,14,15,16]

    # multiplied by
    # [1,0,0,0,
    #  0,0,0,0,
    #  0,0,0,0,
    #  0,0,0,0,]

    # shall be
    # [0,0,0,0,
    #  0,1,2,3,
    #  0,5,6,7,
    #  0,9,10,11]


    H=4
    W=4
    ifm_shape = (1,1,H,W)
    ifm = list(range(1, H*W + 1))

    wgt_shape = (1,1,3,3)

    for r0 in range(3):
      for s0 in range(3):
        wgt = [0]*9
        wgt[r0*3 + s0] = 1

        got, shape = conv2d(ifm,ifm_shape,wgt,wgt_shape,(1,1),(1,1))

        expected = []

        for p in range(H):
          for q in range(W):
            h = p + (r0-1)
            w = q + (s0-1)

            if 0 <= h < H and 0 <= w < W:
              expected.append(ifm[h*W +w])
            else:
              expected.append(0)

        assert got==expected,f'Tap r0={r0}, s0={s0}: expected {expected}, got {got}'


def test_channel_accumulation():
    """Check that convolution reduces over the input channels.

    A 1x1 kernel whose weights are all 1 turns C input channels into one
    output channel that is their elementwise sum. Build the expected result
    with a separate, simpler loop, not by calling conv2d again.

    Raises:
        AssertionError: if the channel reduction is wrong.
    """

    C = 10 # Common to input and filter

    wgt = [1]*C
    wgt_shape = (1,C,1,1)

    N, H, W = 2,3,3

    ifm_shape = (N,C,H,W)
    ifm = random_tensor(ifm_shape,seed=67)

    got, shape = conv2d(ifm, ifm_shape, wgt, wgt_shape)

    assert shape == (N,1,H,W), f'Expected ({N},1,{H},{W}), got {shape}' # Ten input channels turn to one output channel

    expected = [0] * (N*1*H*W) # C input channels reduce to one output channel

    for n in range(N):
      for c in range(C):
        for h in range(H):
          for w in range(W):
            ifm_off = n*C*H*W + c*H*W + h*W + w
            ofm_off = n*1*H*W + 0*H*W + h*W + w # Only one channel, and hence only one index for c
            wgt_off = 0 #it's 1x1 kernel

            expected[ofm_off] += ifm[ifm_off]*wgt[wgt_off]

    assert got == expected, f'expected {expected}, got {got}'


def test_stride_is_a_subsample():
    """Check that a strided output is the stride-1 output sampled every stride.

    With pad=(0, 0), output element (p, q) at stride 2 equals element (2p, 2q)
    at stride 1. Use random_tensor with a fixed seed so a failure repeats.

    Raises:
        AssertionError: if the strided output does not match.
    """

    N, C, H, W = 1, 2, 7, 7
    K, R, S = 3, 3, 3
    ifm_shape = (N,C,H,W)
    wgt_shape = (K,C,R,S)

    ifm = random_tensor(ifm_shape,seed=11)
    wgt = random_tensor(wgt_shape,seed=12)

    out1, shape1 = conv2d(ifm,ifm_shape,wgt,wgt_shape,stride=(1,1),pad=(0,0))
    out2, shape2 = conv2d(ifm,ifm_shape,wgt,wgt_shape,stride=(2,2),pad=(0,0))

    N1,K1,P1,Q1 = shape1
    N2,K2,P2,Q2 = shape2

    for n in range(N):
      for k in range(K):
        for p in range(P2):
          for q in range(Q2):
            off1 = n*K*P1*Q1 + k*P1*Q1 + (2*p)*Q1 + (2*q)
            off2 = n*K*P2*Q2 + k*P2*Q2 + p*Q2 + q

            assert out1[off1] == out2[off2], f'mismatch at (n,k,p,q) = {n},{k},{p},{q}, stride1 out = {out1}, stride2 out = {out2}'


def test_dilation():
    """Check dilation, independent axes, and interactions with padding/stride."""
    got, shape = conv2d(list(range(1, 10)), (1, 1, 3, 3),
                        [1, 2, 3, 4], (1, 1, 2, 2), dilation=(2, 2))
    assert shape == (1, 1, 1, 1), shape
    assert got == [64], got

    got, shape = conv2d(list(range(1, 13)), (1, 1, 3, 4),
                        [1, 2, 3, 4], (1, 1, 2, 2),
                        stride=(2, 1), pad=(1, 0), dilation=(1, 2))
    assert shape == (1, 1, 2, 2), shape
    assert got == [15, 22, 90, 100], got

    # A dilated convolution must equal convolution with a zero-expanded kernel.
    ifm_shape = (2, 2, 5, 7)
    wgt_shape = (3, 2, 2, 3)
    ifm = random_tensor(ifm_shape, seed=31)
    wgt = random_tensor(wgt_shape, seed=32)
    expanded_shape = (3, 2, 3, 7)
    expanded = [0] * num_elements(expanded_shape)
    for k in range(3):
        for c in range(2):
            for r in range(2):
                for s in range(3):
                    expanded[flat_index(expanded_shape, k, c, r*2, s*3)] = \
                        wgt[flat_index(wgt_shape, k, c, r, s)]
    got, shape = conv2d(ifm, ifm_shape, wgt, wgt_shape,
                        stride=(2, 1), pad=(1, 2), dilation=(2, 3))
    expected, expected_shape = conv2d(ifm, ifm_shape, expanded, expanded_shape,
                                     stride=(2, 1), pad=(1, 2))
    assert shape == expected_shape, (shape, expected_shape)
    assert got == expected, (got, expected)

    for dilation in ((0, 1), (1, 0), (-1, 1), (1, -1), (1.5, 1), (1, 1.5)):
        try:
            conv2d([1], (1, 1, 1, 1), [1], (1, 1, 1, 1), dilation=dilation)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for dilation={dilation}")

    try:
        conv2d([1]*9, (1, 1, 3, 3), [1]*4, (1, 1, 2, 2), dilation=(3, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when dilated kernel does not fit")


def run_tests():
    """Run every test and report the result.

    Returns:
        True if all tests pass, False otherwise.
    """
    tests = [
        test_output_size,
        test_known_case,
        test_shift_positions,
        test_channel_accumulation,
        test_stride_is_a_subsample,
        test_dilation,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failures += 1
            print("FAIL %-28s %s" % (t.__name__, exc))
        else:
            print("ok   %s" % t.__name__)
    print("%d/%d passed" % (len(tests) - failures, len(tests)))
    return failures == 0


if __name__ == "__main__":
    run_tests()