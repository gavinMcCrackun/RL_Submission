import numpy as np


def _bresenhamline_nslope(slope):
    # Normalize slope for Bresenham's line algorithm.
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

# Returns npts lines of length max_iter each. (npts x max_iter x dimension)
#     max_iter: Max points to traverse. if -1, maximum number of required
#               points are traversed
def _bresenhamlines(start, end, max_iter):

    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)


def bresenhamline(start, end, max_iter=-1):
    # 
    # Returns a list of points from (start, end] by ray tracing a line b/w the
    # points.
    # Parameters:
    #     start: An array of start points (number of points x dimension)
    #     end:   An end points (1 x dimension)
    #         or An array of end point corresponding to each start point
    #             (number of points x dimension)
    #     max_iter: Max points to traverse. if -1, maximum number of required
    #               points are traversed
    # Returns:
    #     linevox (n x dimension) A cumulative array of all points traversed by
    #     all the lines so far.
    # # Return the points as a single array
    # 
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


# if __name__ == "__main__":
#     s = np.array([[0, 0]])
#     end = np.array([[2, 5]])
    # [[0 1]
    #  [1 2]
    #  [1 3]
    #  [2 4]
    #  [2 5]]
    # print(bresenhamline(s, end))
    # import doctest
    #
    # doctest.testmod()



