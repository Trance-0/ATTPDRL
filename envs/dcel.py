"""
This is the class of helper functions for the DCEL data structure.

Mainly used to show off what I studied over the CSE546 computational geometry course.

Main reference:

Computational Geometry: Algorithms and Applications (the 3M) https://link.springer.com/book/10.1007/978-3-540-77974-2
by Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars.

Chapter 2: Line Segment Intersection

Some functions included:

- compute map overlays for constructed polygons.
"""
import collections
from heapq import heappush, heappop
import math
from sortedcontainers import SortedList

class HalfEdge:
    def __init__(self, origin_vertex):
        """
        Initialize a half-edge.
        Arguments:
            origin_vertex: the origin vertex of the half-edge
        """
        self.origin_vertex = origin_vertex
        # the twin half-edge
        self.twin_half_edge = None
        # the next half-edge along the face boundary
        self.next_half_edge = None
        # the previous half-edge along the face boundary
        self.prev_half_edge = None
        # identity for the face on the left side of the edge, usually none.
        # self.face = None
    
    def destination_vertex(self):
        """
        Return the destination vertex of the half-edge.
        Returns:
            The destination vertex of the half-edge.
        """
        return self.twin_half_edge.origin_vertex

def is_on_segment(point:tuple[float,float],segment:tuple[tuple[float,float],tuple[float,float]],tolerance:float=1e-6) -> bool:
    """
    Check if the point is on the segment.
    Arguments:
        point: a tuple of (x,y)
        segment: a tuple of ((x1,y1),(x2,y2))
        tolerance: a float of the tolerance for the distance between the point and the segment.
    Returns:
        True if the point is on the segment, False otherwise.
    """
    # solve algebraically
    (x1,y1),(x2,y2) = segment
    a=(y2-y1)/(x2-x1)
    c=y1-a*x1
    return abs(a*point[0]+c-point[1]) < tolerance and min(x1,x2) <= point[0] <= max(x1,x2) and min(y1,y2) <= point[1] <= max(y1,y2)
    
def intersect(segment1,segment2) -> tuple[float,float] | None:
    """
    Compute the intersection of two line segments.
    Arguments:
        segment1: a tuple of ((x1,y1),(x2,y2))
        segment2: a tuple of ((x3,y3),(x4,y4))
    Returns:
        The intersection point of the two line segments.
        None if the two line segments are parallel.
    """
    assert type(segment1) == tuple and type(segment2) == tuple and len(segment1) == 2 and len(segment2) == 2
    (x1,y1),(x2,y2) = segment1
    (x3,y3),(x4,y4) = segment2
    denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if denominator == 0:
        return None
    x = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denominator
    y = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denominator
    # check if the intersection is on both segments
    if not is_on_segment((x,y),segment1) or not is_on_segment((x,y),segment2):
        return None
    return x,y

def negative_x_axis_angle(start:tuple[float,float],end:tuple[float,float]) -> float:
    """
    Compute the angle of the segment (start,end) relative to the negative x axis.
    Arguments:
        start: a tuple of (x,y)
        end: a tuple of (x,y)
    Returns:
        The angle of the segment (start,end) relative to the negative x axis.
    """
    # use the lower, left most as start point
    if start[1] < end[1] or (start[1] == end[1] and start[0] < end[0]):
        start,end = end,start
    return -math.atan2(end[1]-start[1],end[0]-start[0])

def clean_segments(segments:list) -> list:
    """
    Clean the segments, remove duplicates and sort by start point
    Arguments:
        segments: a list of line segments, each segment is a tuple of (start,end)
    Returns:
        A list of clean segments.
    """ 
    # clean the segments, remove duplicates and sort by start point
    # default set lower left vertex to be the starting point
    well_oriented_segments = []
    for start,end in segments:
        if start == end:
            raise ValueError(f"Segment {start,end} is a point")
        if start[1] > end[1] or (start[1] == end[1] and start[0] > end[0]):
            well_oriented_segments.append((end,start))
        else:
            well_oriented_segments.append((start,end))
    segments = well_oriented_segments
    # normalize the segments to have the same length, and remove the overlapping segments
    normalized_segments = collections.defaultdict(tuple[float,float])
    for segment in segments:
        start,end = segment
        scale_factor = math.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
        new_end = (start[0]+scale_factor*(end[0]-start[0]),start[1]+scale_factor*(end[1]-start[1]))
        # you may tweak the condition for which segment to keep
        normalized_segments[(start,new_end)] = max(normalized_segments[(start,end)],(start,end))
    segments = list(normalized_segments.values())
    return segments

def line_segment_intersection(segments:list) -> list:
    """
    Report all vertices given the format defined as:
    dict(p:L(p),U(p),C(p))
    where p is the vertex, 
    L(p) is the list of line segments below p, 
    U(p) is the list of line segments above p, 
    C(p) is the list of line segments crossing p.

    Later, we will use this function to extend that to creating the planner subdivision.
    Arguments:
        segments: a list of line segments, each segment is a tuple of (start,end)
    Returns:
        A list of intersection points.
    """
    clean_segments(segments)
    # initialize the vertices dictionary
    vertices = collections.defaultdict(list)
    # store the event points, sorted by increasing y coordinate
    # contents: (y,x,segment,event_type)
    event_queue = []
    # store the intersecting edges with the sweep line, sorted by increasing x, if have same x, then sort by segment angle with -x axis
    # contents: (x,-x axis angle,segment)
    status = SortedList()
    def find_new_events(segment1, segment2, p:tuple[float,float]) -> list:
        """
        Insert new events based on the intersection of two segments and validate by current processing point p
        This function is not supposed to alter the status list.
        Arguments:
            segment1: a tuple of (start,end)
            segment2: a tuple of (start,end)
            p: a tuple of (x,y)
        """
        intersect_x,intersect_y = intersect(segment1,segment2)
        # add new events if 
        # 1. the intersection with the same y coordinate is on the right side of the current processing point
        # 2. the intersection is above the current processing point
        if intersect_x is not None and (intersect_x > p[0] or (intersect_x == p[0] and intersect_y > p[1])):
            # we don't need -x axis angle here because each vertex will be processed together
            heappush(event_queue, (intersect_y,intersect_x,segment1,'intersect'))
            heappush(event_queue, (intersect_y,intersect_x,segment2,'intersect'))
        return []
    def handle_event(event:tuple) -> None:
        """
        Handle an event.
        Arguments:
            event: a tuple of (start,event_type)
        """
        y,x,segment,event_type = event
        # 1. extract all the events containing (x,y)
        Lp,Up,Cp = [],[],[]
        def assign_group(event_type:str,segment:tuple) -> None:
            """
            Assign the segment to the corresponding group.
            Arguments:
                event_type: a string of 'insert', 'remove', 'intersect'
                segment: a tuple of (start,end)
            """
            if event_type == 'insert':
                Lp.append(segment)
            elif event_type == 'remove':
                Up.append(segment)
            elif event_type == 'intersect':
                Cp.append(segment)
            else:
                raise ValueError(f"Invalid event type: {event_type}")
        # assign the current event
        assign_group(event_type,segment)
        # 2. extract all the events containing (x,y)
        while event_queue and event_queue[0][0] == y and event_queue[0][1] == x:
            y,x,segment,event_type = heappop(event_queue)
            assign_group(event_type,segment)
        # 3. update the vertices dictionary
        vertices[x] = Lp,Up,Cp
        # 4. Delete Up (already visited) and Cp (already handled) from status
        for segment in Up:
            start,end = segment
            assert end == (x,y), f"Up segment {segment} is not at ({x},{y})"
            status.discard((start[0],negative_x_axis_angle(start,end),(start,end)))
        for segment in Cp:
            start,end = segment
            assert is_on_segment((x,y),segment), f"Cp segment {segment} is not at ({x},{y})"
            # do not remove since detecting new event did not alter the status list
        # 5. Insert Lp (newly inserted) and Cp (reversing order) into status
        for segment in Lp:
            start,end = segment
            status.add((end[0],negative_x_axis_angle(start,end),(start,end)))
        # 6. Maintain the new events
        # Check if the point is end node only
        # TODO: start here to implement the remaining intersection detection logic
        if len(Up) == 0 and len(Cp) == 0:
            sl,sr=status.bisect_left((x,0)),status.bisect_right((x,0))
            find_new_events(status[sl-1][2],status[sr][2],(x,y))
        else:
            sl,sr=status.bisect_left((x,0)),status.bisect_right((x,0))
            nsl,nsr=status.bisect_left((x,0)),status.bisect_right((x,0))
            find_new_events(status[sl][2],status[sr-1][2],(x,y))
            find_new_events(status[nsl][2],status[nsr-1][2],(x,y))
    # assign insertion events
    for start,end in segments:
        heappush(event_queue, (start[1],start[0],(start,end),'insert'))
        status.add((start[0],negative_x_axis_angle(start,end),(start,end)))
    while event_queue:
        handle_event(heappop(event_queue))
    return vertices


def get_planner_subdivision(segments:list) -> tuple[set,list]:
    """
    Resolve intersections between line segments using plane sweep algorithm.
    Arguments:
        segments: a list of line segments, each segment is a tuple of (x1,y1,x2,y2)
    Returns:
        vertices: a set of vertices (unordered)
        half_edges: a list of half-edges (unordered)
    """
    
    vertices = set()
    half_edges = []
    # algorithm starts here
    # manage all events, sorted by increasing y coordinate, sweeping the edges from bottom to top.
    events = []
    # initialized insert events, sorted by increasing y coordinate, increasing x coordinate and increasing gradient.
    for segment in segments:
        x1,y1,x2,y2 = segment
        gradient = (y2-y1)/(x2-x1)
        heappush(events, (y1,x1,gradient,segment,'insert'))
    # manage all segments intersecting with the sweep line, sorted by their x coordinate
    sweep_line = SortedList()
    while events:
        # each event is a tuple of (y,x,gradient,segments,event_type)
        y,x,gradient,segments,event_type = heappop(events)
        if event_type == 'insert':
            segment=segments[0]
            x1,y1,x2,y2 = segment
            # check position of new segment
            position = sweep_line.bisect_left((y1,segment))
            # insert the segment into the sweep line
            sweep_line.add((y,segment))
            # test if the segment is intersect with the segment at position
            if gradient > 0:
                # the segment is upward, then check lower x segment
                if position == 0: continue
                _,lower_segment = sweep_line[position-1]
                px1,py1,px2,py2 = lower_segment
                intersect_x,intersect_y = intersect((px1,py1,px2,py2),(x1,y1,x2,y2))
                # WARNING: consider the equal case, especially when multiple segments are at the same y coordinate.
                # maybe multiple events should be handled at the same time.
                if intersect_x is not None and intersect_x > px1 and intersect_x < px2:
                    heappush(events, (intersect_y,(segment,lower_segment),'intersect'))
            else:
                # the segment is downward, then check upper x segment
                if position < len(sweep_line)-1:
                    _,upper_segment = sweep_line[position+1]
                    px1,py1,px2,py2 = upper_segment
                    intersect_x,intersect_y = intersect((px1,py1,px2,py2),(x1,y1,x2,y2))
                    if intersect_x is not None and intersect_x > px1 and intersect_x < px2:
                        heappush(events, (intersect_y,(segment,upper_segment),'intersect'))
        elif event_type == 'intersect':
            segment1,segment2 = segments
            px1,py1,px2,py2 = segment1
            px3,py3,px4,py4 = segment2
            intersect_x,intersect_y = intersect((px1,py1,px2,py2),(px3,py3,px4,py4))
            if intersect_x is not None:
                heappush(events, (intersect_y,(segment1,segment2),'insert'))
        elif event_type == 'remove':
            # true segments should only be created on remove events
            # consider cases
            # \     /
            #  \/  /
            #  - - - - -
            #  /\ /
            #  - - - - -
            #    /\
            segment=segments[0]
            # remove the segment from the sweep line
            sweep_line.discard((y,segment))
        else:
            raise ValueError(f"Invalid event type: {event_type}")
    return vertices,half_edges

# simple function tests
if __name__ == "__main__":
    # generate random segments and compute their intersections
    import random
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    n=10
    segments = []
    for i in range(n):
        start = (random.uniform(-10,10),random.uniform(-10,10))
        end = (random.uniform(-10,10),random.uniform(-10,10))
        segments.append((start,end))
    ##########################################################
    # function tests for `intersect` function
    intersections = []
    for i in range(n-1):
        for j in range(i+1,n):
            if intersect(segments[i],segments[j]) is not None:
                intersections.append(intersect(segments[i],segments[j]))
    for segment in segments:
        plt.plot([segment[0][0],segment[1][0]], [segment[0][1],segment[1][1]])
    plt.scatter([x for x,y in intersections], [y for x,y in intersections])
    plt.show()
    ##########################################################
    # # function tests for `negative_x_axis_angle` function
    # angles = []
    # for segment in segments:
    #     angles.append(negative_x_axis_angle(segment[0],segment[1]))
    # for i,segment in enumerate(segments):
    #     plt.plot([segment[0][0],segment[1][0]], [segment[0][1],segment[1][1]],label=f"angle {angles[i]/math.pi*180}°")
    # plt.legend()
    # plt.show()