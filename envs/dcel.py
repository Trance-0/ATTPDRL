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

# class logger configurations
# https://stackoverflow.com/a/56944256
import logging

class ColoredFormatter(logging.Formatter):
    """
    A custom logging formatter that adds color to log messages based on their level.
    """
    # Define ANSI escape codes for different colors
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMAT = "%(asctime)s - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: GREY + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a stream handler and set the custom formatter
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(ColoredFormatter())

# Add the handler to the logger
logger.addHandler(ch)
# class logger ends

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
        # identity for the face on the left side of the edge, usually none, need updated from additional function
        self.face = None
    
    def destination_vertex(self):
        """
        Return the destination vertex of the half-edge.
        Returns:
            The destination vertex of the half-edge.
        """
        return self.twin_half_edge.origin_vertex

def is_on_segment(point:tuple[float,float],segment:tuple[tuple[float,float],tuple[float,float]],tolerance:float=1e-6,exclude_endpoints:bool=True) -> bool:
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
    if x1 == x2:
        if exclude_endpoints:
            return abs(x1-point[0]) < tolerance and min(y1,y2) < point[1] < max(y1,y2)
        else:
            return abs(x1-point[0]) < tolerance and min(y1,y2) <= point[1] <= max(y1,y2)
    a=(y2-y1)/(x2-x1)
    c=y1-a*x1
    if exclude_endpoints:
        return abs(a*point[0]+c-point[1]) < tolerance and min(x1,x2) < point[0] < max(x1,x2) and min(y1,y2) < point[1] < max(y1,y2)
    else:
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
    # TODO: Fix the bug here, some segments are not detected.
    # normalized_segments = collections.defaultdict(tuple[float,float])
    # for segment in segments:
    #     start,end = segment
    #     scale_factor = math.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
    #     new_end = (start[0]+scale_factor*(end[0]-start[0]),start[1]+scale_factor*(end[1]-start[1]))
    #     # you may tweak the condition for which segment to keep
    #     normalized_segments[(start,new_end)] = max(normalized_segments[(start,end)],(start,end))
    # segments = list(normalized_segments.values())
    return segments

def line_segment_intersection(segments:list,edges_only:bool=False) -> dict[tuple[float,float],tuple[list,list,list]]|dict[tuple[float,float],list[tuple[float,float]]]:
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
        edges_only: if True, only return the map of {vertex: vertices belows and connected to the vertex}
    Returns:
        A list of segments (L(p),U(p),C(p))
    """
    segments = clean_segments(segments)
    # initialize the vertices dictionary
    vertices = collections.defaultdict(list)
    # helper dict to store previous intersecting points for each segment, helpful to rebuild partial segments.
    cut_by_segment = collections.defaultdict(list)
    # helper dict to store gradient of each segment, only compute once.
    gradient_by_segment = collections.defaultdict(float)
    # helper dict to restore the source segment for each partial edges.
    source_by_segment = collections.defaultdict(tuple)
    # store the event points, sorted by increasing y coordinate
    # contents: (y,x,segment,event_type) x,y may be coordinates of intersection points.
    event_queue = []
    # store the intersecting edges with the sweep line, sorted by increasing x, if have same x, then sort by segment angle with -x axis
    # contents: (x,-x axis angle,intersecting segment) x may be the coordinate of intersection points.
    # note that intersecting segment is the generated half-edges 
    # (intersecting point or start above, intersecting point or end below). 
    # Status should be event-type-free.
    status = SortedList()
    def print_status() -> str:
        res = "Status:"
        for entry in status:
            res += f"\n Entry: x: {entry[0]}, -x gradient: {entry[1]}, intersecting segment: {entry[2]}"
        return res
    def find_new_events(segment1, segment2, p:tuple[float,float]):
        """
        Insert new events based on the intersection of two segments and validate by current processing point p
        This function is not supposed to alter the status list.
        Arguments:
            segment1: a tuple of (start,end), it is a subsegment of the original segment1
            segment2: a tuple of (start,end), it is a subsegment of the original segment2
            p: a tuple of (x,y)
        """
        sec_p=intersect(segment1,segment2)
        if sec_p:
            intersect_x,intersect_y = sec_p
            # add new events if 
            # 1. the intersection with the same y coordinate is on the right side of the current processing point
            # 2. the intersection is above the current processing point
            if intersect_y < p[1] or (intersect_y==p[1] and intersect_x <= p[0]):
                return
            logger.debug(f"New event point detected: {intersect_x,intersect_y} for edges {segment1} and {segment2} with filter point {p}")
            # divide the segment into two new segments at the intersection point
            new_segment1 = (segment1[0],(intersect_x,intersect_y))
            new_segment2 = (segment2[0],(intersect_x,intersect_y))
            logger.debug(f"Replacing edges {segment1} and {segment2} with {new_segment1} and {new_segment2}")
            # maintain the source_by_segment dict
            source_1 = source_by_segment[segment1]
            source_2 = source_by_segment[segment2]
            source_by_segment[new_segment1] = source_1
            source_by_segment[new_segment2] = source_2
            # checkout stale status list (insert, remove) and insert the current truncated segments back
            status.discard((segment1[0][0],gradient_by_segment[source_1],segment1))
            status.discard((segment2[0][0],gradient_by_segment[source_2],segment2))
            status.add((segment1[0][0],gradient_by_segment[source_1],new_segment1))
            status.add((segment2[0][0],gradient_by_segment[source_2],new_segment2))
            # we don't need -x axis angle here because each vertex will be processed together
            
            logger.debug(f"Add intersecting events for: {intersect_x,intersect_y} for edges {segment1} and {segment2}")
            heappush(event_queue, (intersect_y,intersect_x,source_by_segment[segment1],'intersect'))
            heappush(event_queue, (intersect_y,intersect_x,source_by_segment[segment2],'intersect'))
        else:
            logger.debug(f"No intersection detected for edges {segment1} and {segment2}")
    def handle_event(event:tuple) -> None:
        """
        Handle an event.
        Arguments:
            event: a tuple of (start,event_type)
        """
        y,x,segment,event_type = event
        # 1. extract all the events containing (x,y)
        Lp,Up,Cp = set(),set(),set()
        def assign_group(event_type:str,segment:tuple) -> None:
            """
            Assign the segment to the corresponding group.
            Arguments:
                event_type: a string of 'insert', 'remove', 'intersect'
                segment: a tuple of (start,end)
            """
            if event_type == 'insert':
                Lp.add(segment)
            elif event_type == 'remove':
                Up.add(segment)
            elif event_type == 'intersect':
                Cp.add(segment)
            else:
                raise ValueError(f"Invalid event type: {event_type}")
        # assign the current event
        logger.info(f"Capture initial event: y: {y}, x: {x}, segment: {segment}, event_type: {event_type}")
        assign_group(event_type,segment)
        # 2. extract all the events containing (x,y)
        while event_queue and event_queue[0][0] == y and event_queue[0][1] == x:
            y,x,segment,event_type = heappop(event_queue)
            logger.info(f"Capture additional event: {y,x,segment,event_type}")
            assign_group(event_type,segment)
        # 3. update the vertices dictionary
        assert vertices[(x,y)] == [], f"Vertex {(x,y)} is not empty, current value is {vertices[(x,y)]}, this indicates that a event that supposed to be deleted create a event at the same point. Please try to exclude the end points for detecting intersections or resolve the log files."
        if edges_only:
            # resolve vertices to Up, Cp only with full edges
            cut_Up=[cut_by_segment[segment][-1] for segment in Up]
            cut_Cp=[cut_by_segment[segment][-1] for segment in Cp]
            vertices[(x,y)] = cut_Up+cut_Cp
        else:
            vertices[(x,y)] = list(Lp),list(Up),list(Cp)
        logger.info(f"Event registration done, AsVertices: {(x,y)}, categories: {Lp,Up,Cp}")
        # 4. Delete Up (already visited) and Cp (already handled) from status
        for segment in Up:
            logger.debug(f"Discard Up segment: {segment}")
            _,end = segment
            assert end == (x,y), f"Up segment {segment} is not at ({x},{y})"
            resolve_entry=(cut_by_segment[segment][-1][0],gradient_by_segment[segment],(cut_by_segment[segment][-1],end))
            # assert resolve_entry in status, f"Up segment {segment} is not in status, status is {status}, where resolve_entry is {resolve_entry}, cut_by_segment is {cut_by_segment}, gradient_by_segment is {gradient_by_segment}"
            status.discard(resolve_entry)
        logger.debug(f"Cp segments detected, current cut-points by segment: {cut_by_segment}")
        for segment in Cp:
            logger.debug(f"Discard Cp segment: {segment}, current cut by segment: {cut_by_segment[segment]}")
            assert is_on_segment((x,y),segment), f"Cp segment {segment} is not at ({x},{y})"
            resolve_entry=(cut_by_segment[segment][-1][0],gradient_by_segment[segment],(cut_by_segment[segment][-1],(x,y)))
            # assert resolve_entry in status, f"Cp segment {segment} is not in status, status is {status}, where resolve_entry is {resolve_entry}, cut_by_segment is {cut_by_segment}, gradient_by_segment is {gradient_by_segment}"
            status.discard(resolve_entry)
            # do not remove since detecting new event did not alter the status list
        # 5. Insert Lp (newly inserted) and Cp (reversing order) into status
        # 6.1 record the min and max of newly inserted statuses
        nsl,nsr=None,None
        for segment in Lp:
            # insert cut point
            cut_by_segment[segment].append((x,y))
            logger.debug(f"Insert Lp segment: {segment}")
            start,end = segment
            assert cut_by_segment[segment][-1] == start, f"If you see this error, it means you status management is incorrect, expecting insertion to have prev status to be the start point."
            new_status = (x,gradient_by_segment[segment],(cut_by_segment[segment][-1],end))
            status.add(new_status)
            if nsl is None or nsl > new_status:
                nsl = new_status
            if nsr is None or nsr < new_status:
                nsr = new_status
        for segment in Cp:
            # insert cut point
            cut_by_segment[segment].append((x,y))
            logger.debug(f"Insert Cp segment: {segment}, current cut by segment: {cut_by_segment[segment]}")
            _,end = segment 
            new_status = (x,gradient_by_segment[segment],(cut_by_segment[segment][-1],end))
            status.add(new_status)
            if nsl is None or nsl > new_status:
                nsl = new_status
            if nsr is None or nsr < new_status:
                nsr = new_status
        # 6. Maintain the new events
        logger.debug(f"New status list after operation for sweeping line at ({x},{y}):")
        logger.debug(print_status())
        logger.debug(f"Finding new events... nsl: {nsl}, nsr: {nsr}")
        # Check if the point is end node only
        if nsl is None:
            # in this case, pop event must occurs. And no entry with x left.
            sl_index=status.bisect_left((x,0,((0,0),(0,0))))
            if sl_index >0 and sl_index < len(status):
                sl,sr=status[sl_index-1],status[sl_index]
                find_new_events(sl[2],sr[2],(x,y))
        else:
            # retrieve the entries of inserted statuses
            sl_index,sr_index=status.bisect_left(nsl),status.bisect_left(nsr)
            logger.info(f'resolve sl_index: {sl_index}, sr_index: {sr_index}, status size: {len(status)}')
            if sl_index > 0:
                find_new_events(status[sl_index-1][2],status[sl_index][2],(x,y))
            if sr_index < len(status)-1:
                find_new_events(status[sr_index][2],status[sr_index+1][2],(x,y))
    # assign insertion events
    for start,end in segments:
        heappush(event_queue, (start[1],start[0],(start,end),'insert'))
        gradient_by_segment[(start,end)] = negative_x_axis_angle(start,end)
        # assign source segment for self.
        source_by_segment[(start,end)] = (start,end)
        heappush(event_queue, (end[1],end[0],(start,end),'remove'))
    # main function, easy huh?
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
    vertices=collections.defaultdict(HalfEdge)
    half_edges=[]
    result = [(k,v) for k,v in line_segment_intersection(segments,edges_only=True).items()]
    result.sort(key=lambda x:x[0][1])
    logger.debug(f"received decompositions: {result}")
    # build planner subdivision for each vertex, all recorded points belows the vertex
    for k,v in result:
        if not v: continue
        lower_edges=sorted(v,key=lambda x:-negative_x_axis_angle(k,x))
        logger.debug(f"building edges for vertex {k}, sorted lower edges: {lower_edges}")
        for nv in lower_edges:
            forward=HalfEdge(nv)
            backward=HalfEdge(k)
            # connect two edges
            forward.twin_half_edge=backward
            backward.twin_half_edge=forward
            # resolve edge order for nv
            nv_prev=vertices[nv] if vertices[nv] else forward
            nv_prev.next_half_edge=forward
            forward.prev_half_edge=nv_prev
            vertices[nv]=forward
            # resolve edge order for k
            k_prev=vertices[k] if vertices[k] else backward
            k_prev.next_half_edge=backward
            backward.prev_half_edge=k_prev
            vertices[k]=backward
            # record half-edges
            half_edges.append(forward)
            half_edges.append(backward)
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
        
    # special non-trivial test case:
    segments = [((3.0,1.0),(5.0,5.0)),
                ((4.0,0.0),(4.0,5.0)),
                ((6.0,1.0),(4.0,3.0)),
                ((7.0,1.0),(8.0,4.0)),
                ((2.0,3.0),(1.0,5.0)),
                ((4.0,3.0),(3.0,5.0)),
                ((4.0,3.0),(0.0,5.0))]
    n=len(segments)
    ##########################################################
    # function tests for `intersect` function, brute force n^2 algorithm
    # intersections = []
    # for i in range(n-1):
    #     for j in range(i+1,n):
    #         if intersect(segments[i],segments[j]) is not None:
    #             intersections.append(intersect(segments[i],segments[j]))
    # for segment in segments:
    #     plt.plot([segment[0][0],segment[1][0]], [segment[0][1],segment[1][1]])
    # plt.scatter([x for x,y in intersections], [y for x,y in intersections])
    # plt.show()
    ##########################################################
    # # function tests for `negative_x_axis_angle` function
    # angles = []
    # for segment in segments:
    #     angles.append(negative_x_axis_angle(segment[0],segment[1]))
    # for i,segment in enumerate(segments):
    #     plt.plot([segment[0][0],segment[1][0]], [segment[0][1],segment[1][1]],label=f"angle {angles[i]/math.pi*180}°")
    # plt.legend()
    # plt.show()
    ##########################################################
    # function tests for `line_segment_intersection` function, expected O(nlogn) algorithm
    # for segment in segments:
    #     plt.plot([segment[0][0],segment[1][0]], [segment[0][1],segment[1][1]],color='blue')
    # vertices = line_segment_intersection(segments)
    # for k,v in vertices.items():
    #     # plot intersecting segments only
    #     for segment in v[2]:
    #         plt.plot([segment[0][0],segment[1][0]], [segment[0][1],segment[1][1]],color='red',linestyle='dashed')
    # plt.show()
    ##########################################################
    # function test for 'planner subdivision' function, expected O(nlogn) algorithm
    vertices,half_edges = get_planner_subdivision(segments)