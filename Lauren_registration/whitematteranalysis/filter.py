""" filter.py

Swiss army knife of fiber polydata processing.

This module provides filtering functions that operate on vtkPolyData
objects containing whole-brain tractography.

Preprocess by removing fibers based on length, or endpoint
distance. Downsample. Mask.  Create symmetrized test data.

preprocess
downsample
mask
symmetrize
remove_hemisphere
remove_outliers

"""

import vtk
import numpy

try:
    from joblib import Parallel, delayed
    USE_PARALLEL = 1
except ImportError:
    USE_PARALLEL = 0
    print "<filter.py> Failed to import joblib, cannot multiprocess."
    print "<filter.py> Please install joblib for this functionality."

import fibers
import similarity

verbose = 0


def preprocess(inpd, min_length_mm,
               remove_u=False,
               remove_u_endpoint_dist=40,
               remove_brainstem=False,
               return_indices=False):
    """Remove low-quality fibers.

    Based on fiber length, and optionally on distance between
    endpoints (u-shape has low distance), and inferior location
    (likely in brainstem).

    """

    # set up processing and output objects
    ptids = vtk.vtkIdList()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    inpoints = inpd.GetPoints()
    outpd.SetPoints(inpoints)
    
    # min_length_mm is in mm. Convert to minimum points per fiber
    # by measuring step size (using first two points on first line that has >2 points)
    cell_idx = 0
    # loop over lines until we find one that has more than 2 points
    inpd.GetLines().InitTraversal()
    while (ptids.GetNumberOfIds() < 2) & (cell_idx < inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)
        ##    inpd.GetLines().GetCell(cell_idx, ptids)
        ## the GetCell function is not wrapped in Canopy python-vtk
        cell_idx += 1
        
    # make sure we have some trajectories
    assert ptids.GetNumberOfIds() >= 2

    point0 = inpoints.GetPoint(ptids.GetId(0))
    point1 = inpoints.GetPoint(ptids.GetId(1))
    step_size = numpy.sqrt(numpy.sum(numpy.power(
                numpy.subtract(point0, point1), 2)))
    min_length_pts = round(min_length_mm / step_size)
    print "<filter.py> Minimum length", min_length_mm, \
        "mm. Tractography step size * minimum number of points =", step_size, "*", min_length_pts, ")"

    # loop over lines
    inpd.GetLines().InitTraversal()
    outlines.InitTraversal()

    # keep track of the lines we have kept
    line_indices = list()
    
    for lidx in range(0, inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)

        # first figure out whether to keep this line
        fiber_mask = False
        # test for line being long enough
        if ptids.GetNumberOfIds() > min_length_pts:
            fiber_mask = True

            if remove_u | remove_brainstem:
                # find first and last points on the fiber
                ptid = ptids.GetId(0)
                point0 = inpoints.GetPoint(ptid)
                ptid = ptids.GetId(ptids.GetNumberOfIds() - 1)
                point1 = inpoints.GetPoint(ptid)

            if remove_u:
                # compute distance between endpoints
                endpoint_dist = numpy.sqrt(numpy.sum(numpy.power(
                            numpy.subtract(point0, point1), 2)))
                if endpoint_dist < remove_u_endpoint_dist:
                    fiber_mask = False

            if remove_brainstem:
                # compute average SI (third coordinate) < -40
                mean_sup_inf = (point0[2] + point1[2]) / 2
                if mean_sup_inf < -40:
                    fiber_mask = False

        # if we are keeping the line do it
        if fiber_mask:
            outlines.InsertNextCell(ptids)
            line_indices.append(lidx)
            if verbose:
                if lidx % 100 == 0:
                    print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()

    outpd.SetLines(outlines)

    if verbose:
        msg = outpd.GetNumberOfLines(), "/", inpd.GetNumberOfLines()
        print "<filter.py> Number of lines:", msg

    if return_indices:
        return outpd, numpy.array(line_indices)
    else:
        return outpd

def downsample(inpd, output_number_of_lines, return_indices=False, preserve_point_data=False):
    """ Random (down)sampling of fibers without replacement. """

    num_lines = inpd.GetNumberOfLines()

    if num_lines < output_number_of_lines:
        return inpd

    # randomly pick the lines that we will keep
    line_indices = numpy.random.permutation(num_lines - 1)
    line_indices = line_indices[0:output_number_of_lines]
    fiber_mask = numpy.zeros(num_lines)
    fiber_mask[line_indices] = 1

    # don't color by line index by default, preserve whatever was there
    #outpd = mask(inpd, fiber_mask, fiber_mask)
    outpd = mask(inpd, fiber_mask, preserve_point_data=preserve_point_data)

    # final line count
    #print "<filter.py> Number of lines selected:", outpd.GetNumberOfLines()
    if return_indices:
        # return sorted indices, this is the line ordering of output
        # polydata (because we mask rather than changing input line order)
        return outpd, numpy.sort(line_indices)
    else:
        return outpd


def mask(inpd, fiber_mask, color=None, preserve_point_data=False):
    """ Keep lines and their points where fiber_mask == 1.

     Unlike vtkMaskPolyData that samples every nth cell, this function
     uses an actual mask, and also gets rid of points that
     are not used by any cell, reducing the size of the polydata file.

     This code also sets scalar cell data to input color data.  This
     input data is expected to be 1 or 3 components.

     If there is no input cell scalar color data, existing cell
     scalars that we have created (EmbeddingColor, ClusterNumber,
     EmbeddingCoordinate) are looked for and masked as well.

     """

    inpoints = inpd.GetPoints()
    inpointdata = inpd.GetPointData()
    
    # output and temporary objects
    ptids = vtk.vtkIdList()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    outpoints = vtk.vtkPoints()
    outcolors = None
    outpointdata = outpd.GetPointData()

    if color is not None:
        # if input is RGB
        if len(color.shape) == 2:
            if color.shape[1] == 3:
                outcolors = vtk.vtkUnsignedCharArray()
                outcolors.SetNumberOfComponents(3)

        # otherwise output floats as colors
        if outcolors == None:
            outcolors = vtk.vtkFloatArray()
    else:
        # this is really specific to our code, perhaps there is a more
        # general way
        if inpd.GetCellData().GetNumberOfArrays() > 0:
            # look for our arrays by name
            if verbose:
                print "<filter.py> looking for arrays"
            if inpd.GetCellData().GetArray('ClusterNumber'):
                array = vtk.vtkIntArray()
                array.SetName('ClusterNumber')
                outpd.GetCellData().AddArray(array)
                # this will be active unless we have embedding colors
                outpd.GetCellData().SetActiveScalars('ClusterNumber')
            if inpd.GetCellData().GetArray('EmbeddingColor'):
                array = vtk.vtkUnsignedCharArray()
                array.SetNumberOfComponents(3)
                array.SetName('EmbeddingColor')
                outpd.GetCellData().AddArray(array)
                outpd.GetCellData().SetActiveScalars('EmbeddingColor')
                print "<filter.py> added array embed color"
            if inpd.GetCellData().GetArray('EmbeddingCoordinate'):
                array = vtk.vtkFloatArray()
                ncomp = inpd.GetCellData().GetArray('EmbeddingCoordinate').GetNumberOfComponents()
                array.SetNumberOfComponents(ncomp)
                array.SetName('EmbeddingCoordinate')
                outpd.GetCellData().AddArray(array)

    #check for point data arrays to keep
    if preserve_point_data:
        if inpointdata.GetNumberOfArrays() > 0:
            point_data_array_indices = range(inpointdata.GetNumberOfArrays())            
            for idx in point_data_array_indices:
                array = inpointdata.GetArray(idx)
                out_array = vtk.vtkFloatArray()
                out_array.SetNumberOfComponents(array.GetNumberOfComponents())
                out_array.SetName(array.GetName())
                outpointdata.AddArray(out_array)
        else:
            preserve_point_data = False
                
    # loop over lines
    inpd.GetLines().InitTraversal()
    outlines.InitTraversal()

    for lidx in range(0, inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)

        if fiber_mask[lidx]:

            if verbose:
                if lidx % 100 == 0:
                    print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()

            # get points for each ptid and add to output polydata
            cellptids = vtk.vtkIdList()

            for pidx in range(0, ptids.GetNumberOfIds()):
                point = inpoints.GetPoint(ptids.GetId(pidx))
                idx = outpoints.InsertNextPoint(point)
                cellptids.InsertNextId(idx)
                if preserve_point_data:
                    for idx in point_data_array_indices:
                        array = inpointdata.GetArray(idx)
                        out_array = outpointdata.GetArray(idx)
                        out_array.InsertNextTuple(array.GetTuple(ptids.GetId(pidx)))

            outlines.InsertNextCell(cellptids)

                    

            if color is not None:
                # this code works with either 3 or 1 component only
                if outcolors.GetNumberOfComponents() == 3:
                    outcolors.InsertNextTuple3(color[lidx,0], color[lidx,1], color[lidx,2])
                else:
                    outcolors.InsertNextTuple1(color[lidx])
            else:
                if outpd.GetCellData().GetArray('EmbeddingColor'):
                    outpd.GetCellData().GetArray('EmbeddingColor').InsertNextTuple(inpd.GetCellData().GetArray('EmbeddingColor').GetTuple(lidx))
                if outpd.GetCellData().GetArray('ClusterNumber'):
                    outpd.GetCellData().GetArray('ClusterNumber').InsertNextTuple(inpd.GetCellData().GetArray('ClusterNumber').GetTuple(lidx))
                if outpd.GetCellData().GetArray('EmbeddingCoordinate'):
                    outpd.GetCellData().GetArray('EmbeddingCoordinate').InsertNextTuple(inpd.GetCellData().GetArray('EmbeddingCoordinate').GetTuple(lidx))

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    if color is not None:
        outpd.GetCellData().SetScalars(outcolors)

    print "<filter.py> Fibers sampled:", outpd.GetNumberOfLines(), "/", inpd.GetNumberOfLines()

    return outpd


def symmetrize(inpd):
    """Generate symmetric polydata by reflecting.

    Output polydata has twice as many lines as input.

    """

    # output and temporary objects
    ptids = vtk.vtkIdList()
    points = inpd.GetPoints()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    outpoints = vtk.vtkPoints()
    outpoints.DeepCopy(points)

    # loop over lines
    inpd.GetLines().InitTraversal()
    outlines.InitTraversal()

    # set scalar cell data to 1 for orig, -1 for reflect, for vis
    outcolors = vtk.vtkFloatArray()

    # index into end of point array
    lastidx = outpoints.GetNumberOfPoints()
    print "<filter.py> Input number of points: ", lastidx

    # loop over all lines, insert line and reflected copy into output pd
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        if verbose:
            if lidx % 100 == 0:
                print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()

        inpd.GetLines().GetNextCell(ptids)

        num_points = ptids.GetNumberOfIds()

        # insert fiber (ptids are same since new points go at the end)
        outlines.InsertNextCell(ptids)
        outcolors.InsertNextTuple1(1)

        # insert reflection into END of point array and into line array
        refptids = vtk.vtkIdList()
        for pidx in range(0, num_points):

            point = points.GetPoint(ptids.GetId(pidx))

            # reflect (RAS -> reflect first value)
            refpoint = (-point[0], point[1], point[2])
            idx = outpoints.InsertNextPoint(refpoint)
            refptids.InsertNextId(idx)

        outlines.InsertNextCell(refptids)
        outcolors.InsertNextTuple1(-1)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    outpd.GetCellData().SetScalars(outcolors)

    return outpd


def remove_hemisphere(inpd, hemisphere=-1):
    """ Remove left (-1) or right (+1) hemisphere points. """

    # output and temporary objects
    ptids = vtk.vtkIdList()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    outlines.InitTraversal()
    outpoints = vtk.vtkPoints()

    # loop over lines
    inpd.GetLines().InitTraversal()

    # loop over all lines, inserting into output only
    # the part in the correct hemisphere
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        if verbose:
            if lidx % 100 == 0:
                print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()

        inpd.GetLines().GetNextCell(ptids)

        num_points = ptids.GetNumberOfIds()

        # insert kept points into point array and into line array
        keptptids = vtk.vtkIdList()
        for pidx in range(0, num_points):

            point = inpd.GetPoints().GetPoint(ptids.GetId(pidx))

            # if we keep this point (if not in removed hemisphere)
            if ((point[0] < 0) &
                (hemisphere == 1)) | ((point[0] > 0) &
                                      (hemisphere == -1)):

                idx = outpoints.InsertNextPoint(point)
                keptptids.InsertNextIdx(idx)

        outlines.InsertNextCell(keptptids)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)

    return outpd


def remove_outliers(inpd, min_fiber_distance, n_jobs=2):
    """ Remove fibers that have no other nearby fibers, i.e. outliers.

    The pairwise fiber distance matrix is computed, then fibers
    are rejected if their average neighbor distance (using top 3
    neighbors) is higher than min_fiber_distance.

    """

    fiber_array = fibers.FiberArray()
    fiber_array.points_per_fiber = 5
    fiber_array.convert_from_polydata(inpd)

    fiber_indices = range(0, fiber_array.number_of_fibers)

    sigmasq = 10 * 10

    # pairwise distance matrix
    if USE_PARALLEL:
        distances = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(similarity.fiber_distance)(
                fiber_array.get_fiber(lidx),
                fiber_array,
                0)
            for lidx in fiber_indices)

        distances = numpy.array(distances)

    else:
        distances = numpy.zeros((fiber_array.number_of_fibers, fiber_array.number_of_fibers))
        for lidx in fiber_indices:
            distances[lidx, :] = \
                similarity.fiber_distance(fiber_array.get_fiber(lidx), fiber_array, 0)

    # now we check where there are no nearby fibers in d
    fiber_mask = numpy.ones(fiber_array.number_of_fibers)
    mindist = numpy.zeros(fiber_array.number_of_fibers)
    for lidx in fiber_indices:
        dist = numpy.sort(distances[lidx, :])
        mindist[lidx] = (dist[1] + dist[2] + dist[3]) / 3
        #print mindist[lidx], dist[-1]
        #fiber_mask[lidx] = (mindist[lidx] < 5)

    # keep only fibers who have nearby similar fibers
    fiber_mask = mindist < min_fiber_distance

    if True:
        num_fibers = len(numpy.nonzero(fiber_mask)[0]), "/", len(fiber_mask)
        print "<filter.py> Number retained after outlier removal: ", num_fibers

    outpd = mask(inpd, fiber_mask, mindist)

    return outpd

def smooth(inpd, fiber_distance_sigma = 25, points_per_fiber=30, n_jobs=2, upper_thresh=30):
    """ Average nearby fibers.
    
    The pairwise fiber distance matrix is computed, then fibers
    are averaged with their neighbors using Gaussian weighting.

    The "local density" or soft neighbor count is also output.
    """

    sigmasq = fiber_distance_sigma * fiber_distance_sigma
    
    # polydata to array conversion, fixed-length fiber representation
    current_fiber_array = fibers.FiberArray()
    current_fiber_array.points_per_fiber = points_per_fiber
    current_fiber_array.convert_from_polydata(inpd)

    # fiber list data structure initialization for easy fiber averaging
    curr_count = list()
    curr_fibers = list()
    next_fibers = list()
    next_weights = list()
    for lidx in range(0, current_fiber_array.number_of_fibers):
        curr_fibers.append(current_fiber_array.get_fiber(lidx))
        curr_count.append(1)

    fiber_indices = range(0, current_fiber_array.number_of_fibers)

    print "<filter.py> Computing pairwise distances..."
    
    # pairwise distance matrix
    if USE_PARALLEL:
        distances = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(similarity.fiber_distance)(
            current_fiber_array.get_fiber(lidx),
            current_fiber_array,
            0, 'Hausdorff')
            for lidx in fiber_indices)
        distances = numpy.array(distances)
    else:
        distances = \
            numpy.zeros(
            (current_fiber_array.number_of_fibers,
             current_fiber_array.number_of_fibers))
        for lidx in fiber_indices:
            distances[lidx, :] = \
                similarity.fiber_distance(
                    current_fiber_array.get_fiber(lidx),
                    current_fiber_array, 0)
   
    # gaussian smooth all fibers using local neighborhood
    for fidx in fiber_indices:
        if (fidx % 100) == 0:
            print fidx, '/', current_fiber_array.number_of_fibers

        # find indices of all nearby fibers
        indices = numpy.nonzero(distances[fidx] < upper_thresh)[0]
        local_fibers = list()
        local_weights = list()

        for idx in indices:
            dist = distances[fidx][idx]
            weight = numpy.exp(-(dist*dist)/sigmasq)
            local_fibers.append(curr_fibers[idx] * weight)
            local_weights.append(weight)
        # actually perform the weighted average
        # start with the one under the center of the kernel
        #out_fiber = curr_fibers[fidx]
        #out_weights = 1.0
        out_fiber = local_fibers[0]
        out_weights = local_weights[0]
        for fiber in local_fibers[1:]:
            out_fiber += fiber
        for weight in local_weights[1:]:
            out_weights += weight
        out_fiber = out_fiber / out_weights
        next_fibers.append(out_fiber)
        next_weights.append(out_weights)

    # set up array for output
    output_fiber_array = fibers.FiberArray()    
    output_fiber_array.number_of_fibers = len(curr_fibers)
    output_fiber_array.points_per_fiber = points_per_fiber
    dims = [output_fiber_array.number_of_fibers, output_fiber_array.points_per_fiber]
    # fiber data
    output_fiber_array.fiber_array_r = numpy.zeros(dims)
    output_fiber_array.fiber_array_a = numpy.zeros(dims)
    output_fiber_array.fiber_array_s = numpy.zeros(dims)
    next_fidx = 0
    for next_fib in next_fibers:
        output_fiber_array.fiber_array_r[next_fidx] = next_fib.r
        output_fiber_array.fiber_array_a[next_fidx] = next_fib.a
        output_fiber_array.fiber_array_s[next_fidx] = next_fib.s
        next_fidx += 1
            
    # convert output to polydata
    outpd = output_fiber_array.convert_to_polydata()
    
    # color by the weights or "local density"
    # color output by the number of fibers that each output fiber corresponds to
    outcolors = vtk.vtkFloatArray()
    outcolors.SetName('KernelDensity')
    for weight in next_weights:
        outcolors.InsertNextTuple1(weight)
    #outpd.GetCellData().SetScalars(outcolors)
    outpd.GetCellData().AddArray(outcolors)
    outpd.GetCellData().SetActiveScalars('KernelDensity')

    return outpd, numpy.array(next_weights)
    
def anisotropic_smooth(inpd, fiber_distance_threshold, points_per_fiber=30, n_jobs=2):
    """ Average nearby fibers.
    
    The pairwise fiber distance matrix is computed, then fibers
    are averaged with their neighbors until an edge (>max_fiber_distance) is encountered.

    """

    # polydata to array conversion, fixed-length fiber representation
    current_fiber_array = fibers.FiberArray()
    current_fiber_array.points_per_fiber = points_per_fiber
    current_fiber_array.convert_from_polydata(inpd)
    original_number_of_fibers = current_fiber_array.number_of_fibers
    
    # fiber list data structure initialization for easy fiber averaging
    curr_count = list()
    curr_fibers = list()
    curr_indices = list()
    for lidx in range(0, current_fiber_array.number_of_fibers):
        curr_fibers.append(current_fiber_array.get_fiber(lidx))
        curr_count.append(1)
        curr_indices.append(list([lidx]))
        
    converged = False
    iteration_count = 0
    
    while not converged:
        print "<filter.py> ITERATION:", iteration_count, "SUM FIBER COUNTS:", numpy.sum(numpy.array(curr_count))
        print "<filter.py> number indices", len(curr_indices)
        
        # fiber data structures for output of this iteration
        next_fibers = list()
        next_count = list()
        next_indices = list()
        
        # information for this iteration
        done = numpy.zeros(current_fiber_array.number_of_fibers)
        fiber_indices = range(0, current_fiber_array.number_of_fibers)
    
        # pairwise distance matrix
        if USE_PARALLEL:
            distances = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(similarity.fiber_distance)(
                current_fiber_array.get_fiber(lidx),
                current_fiber_array,
                0, 'Hausdorff')
                for lidx in fiber_indices)
            distances = numpy.array(distances)
        else:
            distances = \
                numpy.zeros(
                (current_fiber_array.number_of_fibers,
                 current_fiber_array.number_of_fibers))
            for lidx in fiber_indices:
                distances[lidx, :] = \
                    similarity.fiber_distance(
                        current_fiber_array.get_fiber(lidx),
                        current_fiber_array, 0)

        # distances to self are not of interest
        for lidx in fiber_indices:
            distances[lidx,lidx] = numpy.inf
        
        # sort the pairwise distances. 
        distances_flat = distances.flatten()
        pair_order = numpy.argsort(distances_flat)

        print "<filter.py> DISTANCE MIN:", distances_flat[pair_order[0]], \
            "DISTANCE COUNT:", distances.shape

        # if the smallest distance is greater or equal to the
        # threshold, we have converged
        if distances_flat[pair_order[0]] >= fiber_distance_threshold:
            converged = True
            print "<filter.py> CONVERGED"
            break
        else:
            print "<filter.py> NOT CONVERGED"
            
        # loop variables
        idx = 0
        pair_idx = pair_order[idx]
        number_of_fibers = distances.shape[0]
        number_averages = 0
        
        # combine nearest neighbors unless done, until hit threshold
        while distances_flat[pair_idx] < fiber_distance_threshold:
            # find the fiber indices corresponding to this pairwise distance
            # use div and mod
            f_row = pair_idx / number_of_fibers
            f_col = pair_idx % number_of_fibers

            # check if this neighbor pair can be combined
            combine = (not done[f_row]) and (not done[f_col])
            if combine :
                done[f_row] += 1
                done[f_col] += 1
                # weighted average of the fibers (depending on how many each one represents)
                next_fibers.append(
                    (curr_fibers[f_row] * curr_count[f_row] + \
                     curr_fibers[f_col] *curr_count[f_col]) / \
                    (curr_count[f_row] + curr_count[f_col]))
                # this was the regular average
                #next_fibers.append((curr_fibers[f_row] + curr_fibers[f_col])/2)
                next_count.append(curr_count[f_row] + curr_count[f_col])
                number_averages += 1
                #next_indices.append(list([curr_indices[f_row], curr_indices[f_col]]))
                next_indices.append(list(curr_indices[f_row] + curr_indices[f_col]))
                
            # increment for the loop
            idx += 1
            pair_idx = pair_order[idx]

        # copy through any unvisited (already converged) fibers
        unvisited = numpy.nonzero(done==0)[0]
        for fidx in unvisited:
            next_fibers.append(curr_fibers[fidx])
            next_count.append(curr_count[fidx])
            next_indices.append(curr_indices[fidx])
            
        # set up for next iteration
        curr_fibers = next_fibers
        curr_count = next_count
        curr_indices = next_indices
        iteration_count += 1

        # set up array for next iteration distance computation
        current_fiber_array = fibers.FiberArray()    
        current_fiber_array.number_of_fibers = len(curr_fibers)
        current_fiber_array.points_per_fiber = points_per_fiber
        dims = [current_fiber_array.number_of_fibers, current_fiber_array.points_per_fiber]
        # fiber data
        current_fiber_array.fiber_array_r = numpy.zeros(dims)
        current_fiber_array.fiber_array_a = numpy.zeros(dims)
        current_fiber_array.fiber_array_s = numpy.zeros(dims)
        curr_fidx = 0
        for curr_fib in curr_fibers:
            current_fiber_array.fiber_array_r[curr_fidx] = curr_fib.r
            current_fiber_array.fiber_array_a[curr_fidx] = curr_fib.a
            current_fiber_array.fiber_array_s[curr_fidx] = curr_fib.s
            curr_fidx += 1

        print "<filter.py> SUM FIBER COUNTS:", numpy.sum(numpy.array(curr_count)), "SUM DONE FIBERS:", numpy.sum(done)
        print "<filter.py> MAX COUNT:" , numpy.max(numpy.array(curr_count)), "AVGS THIS ITER:", number_averages

    # when converged, convert output to polydata
    outpd = current_fiber_array.convert_to_polydata()

    # color output by the number of fibers that each output fiber corresponds to
    outcolors = vtk.vtkFloatArray()
    outcolors.SetName('FiberTotal')
    for count in curr_count:
        outcolors.InsertNextTuple1(count)
    outpd.GetCellData().SetScalars(outcolors)

    # also color the input pd by output cluster number
    cluster_numbers = numpy.zeros(original_number_of_fibers)
    cluster_count = numpy.zeros(original_number_of_fibers)
    cluster_idx = 0
    for index_list in curr_indices:
        indices = numpy.array(index_list).astype(int)
        cluster_numbers[indices] = cluster_idx
        cluster_count[indices] = curr_count[cluster_idx]
        cluster_idx += 1
    outclusters =  vtk.vtkFloatArray()
    outclusters.SetName('ClusterNumber')
    for cluster in cluster_numbers:
        outclusters.InsertNextTuple1(cluster)
    inpd.GetCellData().AddArray(outclusters)
    inpd.GetCellData().SetActiveScalars('ClusterNumber')

    return outpd, numpy.array(curr_count), inpd, cluster_numbers, cluster_count
    

    
def laplacian_of_gaussian(inpd, fiber_distance_sigma = 25, points_per_fiber=30, n_jobs=2, upper_thresh=30):
    """ Filter nearby fibers, using LoG weights.
    
    The pairwise fiber distance matrix is computed, then fibers are
    averaged with their neighbors using LoG weighting.  This is
    essentially a fiber subtraction operation, giving vectors pointing
    from the center fiber under the kernel, to all nearby fibers. Thus
    the output of this operation is not a fiber, but we compute
    properties of the output that might be interesting and related to
    fibers. We summarize the result using the average vector at each
    fiber point (output is its magnitude, similar to edge
    strength). The covariance of the vectors is also
    investigated. This matrix would be spherical in an isotropic
    region such as a tract center (tube/line detector), or planar in a
    sheetlike tract (sheet detector).

    The equation is: (1-d^2/sigma^2) exp(-d^2/(2*sigma^2)), and
    weights are normalized in the neighborhood (weighted averaging).
    """

    sigmasq = fiber_distance_sigma * fiber_distance_sigma
    
    # polydata to array conversion, fixed-length fiber representation
    fiber_array = fibers.FiberArray()
    fiber_array.points_per_fiber = points_per_fiber
    fiber_array.convert_from_polydata(inpd)

    fiber_indices = range(0, fiber_array.number_of_fibers)

    # pairwise distance matrix
    if USE_PARALLEL:
        distances = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(similarity.fiber_distance)(
            fiber_array.get_fiber(lidx),
            fiber_array,
            0, 'Hausdorff')
            for lidx in fiber_indices)
        distances = numpy.array(distances)
    else:
        distances = \
            numpy.zeros(
            (fiber_array.number_of_fibers,
             fiber_array.number_of_fibers))
        for lidx in fiber_indices:
            distances[lidx, :] = \
                similarity.fiber_distance(
                    fiber_array.get_fiber(lidx),
                    fiber_array, 0)

    # fiber list data structure initialization for easy fiber averaging
    fiber_list = list()
    for lidx in range(0, fiber_array.number_of_fibers):
        fiber_list.append(fiber_array.get_fiber(lidx))

    filter_vectors = list()
    filter_vector_magnitudes = list()
    filter_confidences = list()
    
    # gaussian smooth all fibers using local neighborhood
    for fidx in fiber_indices:
        if (fidx % 100) == 0:
            print fidx, '/', fiber_array.number_of_fibers

        current_fiber = fiber_list[fidx]

        # find indices of all nearby fibers
        # this includes the center fiber under the kernel
        indices = numpy.nonzero(distances[fidx] < upper_thresh)[0]
        local_fibers = list()
        local_weights = list()

        for idx in indices:
            dist = distances[fidx][idx]
            # compute filter kernel weights
            weight = numpy.exp(-(dist*dist)/sigmasq)
            #weight = (1 - (dist*dist)/sigmasq) * numpy.exp(-(dist*dist)/(2*sigmasq))
            local_fibers.append(fiber_list[idx])
            local_weights.append(weight)

        # actually perform the weighted average
        #mean_weight = numpy.mean(numpy.array(local_weights))
        #out_weights = local_weights[0]
        #for weight in local_weights[1:]:
        #    out_weights += weight
        # the weights must sum to 0 for LoG
        # (response in constant region is 0)
        #mean_weight = out_weights / len(local_weights)
        #local_normed_weights = list()
        #for weight in local_weights:
        #    local_normed_weights.append(weight - mean_weight)

        #match_fiber = local_fibers[0]
        #out_vector = local_fibers[0] * local_normed_weights[0]
        idx = 0
        for fiber in local_fibers:
            #out_vector += fiber
            # ensure fiber ordering by matching to current fiber only
            # otherwise the order is undefined after fiber subtraction
            matched_fiber = current_fiber.match_order(fiber)
            #filtered_fiber = matched_version * local_normed_weights[idx]
            #filtered_fiber = matched_version * local_weights[idx]
            if idx == 0:
                out_vector = fibers.Fiber()
                out_vector.points_per_fiber = points_per_fiber
                out_vector.r = numpy.zeros(points_per_fiber)
                out_vector.a = numpy.zeros(points_per_fiber)
                out_vector.s = numpy.zeros(points_per_fiber)
            #filtered_fiber = match_fiber.match_order(fiber)
            #out_vector.r = (out_vector.r + matched_fiber.r) * local_weights[idx]
            #out_vector.a = (out_vector.a + matched_fiber.a) * local_weights[idx]
            #out_vector.s = (out_vector.s + matched_fiber.s) * local_weights[idx]
            out_vector.r += (current_fiber.r - matched_fiber.r) * local_weights[idx]
            out_vector.a += (current_fiber.a - matched_fiber.a) * local_weights[idx]
            out_vector.s += (current_fiber.s - matched_fiber.s) * local_weights[idx]
            idx += 1

        total_weights = numpy.sum(numpy.array(local_weights))
        out_vector = out_vector / total_weights       

        filter_vectors.append(out_vector)
        filter_confidences.append(total_weights)

        filter_vector_magnitudes.append(numpy.sqrt(\
                numpy.multiply(out_vector.r, out_vector.r) + \
                    numpy.multiply(out_vector.a, out_vector.a) + \
                    numpy.multiply(out_vector.s, out_vector.s)))
        #filter_vector_magnitudes.append(numpy.sum(out_vector.r))


    # output a new pd!!!!
    # with fixed length fibers. and the new vector field.
    # output the vectors from the filtering
    outpd = fiber_array.convert_to_polydata()
    vectors = vtk.vtkFloatArray()
    vectors.SetName('FiberDifferenceVectors')
    vectors.SetNumberOfComponents(3)
    for vec in filter_vectors:
        for idx in range(points_per_fiber):
            vectors.InsertNextTuple3(vec.r[idx],vec.a[idx],vec.s[idx])
    magnitudes = vtk.vtkFloatArray()
    magnitudes.SetName('FiberDifferenceMagnitudes')
    magnitudes.SetNumberOfComponents(1)
    for mag in filter_vector_magnitudes:
        for idx in range(points_per_fiber):
            magnitudes.InsertNextTuple1(mag[idx])
    confidences = vtk.vtkFloatArray()
    confidences.SetName('FiberDifferenceConfidences')
    confidences.SetNumberOfComponents(1)
    for mag in filter_confidences:
        for idx in range(points_per_fiber):
            confidences.InsertNextTuple1(mag)
         
    outpd.GetPointData().AddArray(vectors)
    outpd.GetPointData().SetActiveVectors('FiberDifferenceVectors')

    outpd.GetPointData().AddArray(confidences)
    outpd.GetPointData().SetActiveScalars('FiberDifferenceConfidences')

    outpd.GetPointData().AddArray(magnitudes)
    outpd.GetPointData().SetActiveScalars('FiberDifferenceMagnitudes')

    # color by the weights or "local density"
    # color output by the number of fibers that each output fiber corresponds to
    #outcolors = vtk.vtkFloatArray()
    #outcolors.SetName('KernelDensity')
    #for weight in next_weights:
    #    outcolors.InsertNextTuple1(weight)
    #inpd.GetCellData().AddArray(outcolors)
    #inpd.GetCellData().SetActiveScalars('KernelDensity')
    #outcolors = vtk.vtkFloatArray()
    #outcolors.SetName('EdgeMagnitude')
    #for magnitude in filter_vector_magnitudes:
    #    outcolors.InsertNextTuple1(magnitude)
    #inpd.GetCellData().AddArray(outcolors)
    #inpd.GetCellData().SetActiveScalars('EdgeMagnitude')

    return outpd, numpy.array(filter_vector_magnitudes)

def pd_to_array(inpd, dims=225):
    count_vol = numpy.ndarray([dims,dims,dims])
    ptids = vtk.vtkIdList()
    points = inpd.GetPoints()
    data_vol = []    
    # check for cell data
    cell_data = inpd.GetCellData().GetScalars()
    if cell_data:
        data_vol = numpy.ndarray([dims,dims,dims])
    # loop over lines
    inpd.GetLines().InitTraversal()
    print "<filter.py> Input number of points: ",\
        points.GetNumberOfPoints(),\
        "lines:", inpd.GetNumberOfLines() 
    # loop over all lines
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        #if verbose:
        #    if lidx % 1 == 0:
        #        print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()
        inpd.GetLines().GetNextCell(ptids)
        num_points = ptids.GetNumberOfIds()
        for pidx in range(0, num_points):
            point = points.GetPoint(ptids.GetId(pidx))
            # center so that 0,0,0 moves to 100,100,100
            point = numpy.round(numpy.array(point) + 110)         
            count_vol[point[0], point[1], point[2]] += 1
            if cell_data:
                data_vol[point[0], point[1], point[2]] += cell_data.GetTuple(lidx)[0]
    return count_vol, data_vol
    
def array_to_vtk(inarray, name='from_numpy'):
    vol = vtk.vtkImageData()
    dims = inarray.shape
    vol.SetDimensions(dims[0], dims[1], dims[2])
    vol.SetOrigin(0,0,0)
    #vol.SetSpacing(gridSpacing,gridSpacing,gridSpacing)
    sc = vtk.vtkShortArray()
    sc.SetNumberOfValues(dims[0] * dims[1] * dims[2])
    sc.SetNumberOfComponents(1)
    sc.SetName(name)
    for ii,tmp in enumerate(inarray.flatten()):
	sc.SetValue(ii,round((numpy.abs(tmp))*100))
    vol.GetPointData().SetScalars(sc)
    return vol
    

def measure_line_lengths(inpd):
    ptids = vtk.vtkIdList()
    points = inpd.GetPoints()
    output_lengths = numpy.zeros(inpd.GetNumberOfLines())
    # loop over lines
    inpd.GetLines().InitTraversal()
    print "<filter.py> Input number of points: ",\
        points.GetNumberOfPoints(),\
        "lines:", inpd.GetNumberOfLines() 
    # loop over all lines
    for lidx in range(0, inpd.GetNumberOfLines()):
        # progress
        #if verbose:
        #    if lidx % 1 == 0:
        #        print "<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines()
        inpd.GetLines().GetNextCell(ptids)
        output_lengths[lidx] = ptids.GetNumberOfIds()
    return(output_lengths)
