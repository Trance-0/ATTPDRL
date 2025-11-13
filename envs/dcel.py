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
import numpy as np
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
    
    def __str__(self) -> str:
        return f"HalfEdge({(self.origin_vertex,self.destination_vertex())}, twin_half_edge={(self.twin_half_edge.origin_vertex,self.twin_half_edge.destination_vertex())},next_half_edge={(self.next_half_edge.origin_vertex,self.next_half_edge.destination_vertex())},prev_half_edge={(self.prev_half_edge.origin_vertex,self.prev_half_edge.destination_vertex())},face={self.face})"
    
    def str(self) -> str:
        return f"HalfEdge({(self.origin_vertex,self.destination_vertex())})"

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
        if any(start == end for start,end in well_oriented_segments):
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
        vertices: a dict of vertices, with maximum angle to -x-axis, pointing outwards
        half_edges: a list of half-edges (unordered)
    """
    # record associated index for half-edges, and parse them in the end
    # update on the fly cause some error for assigning half-edges
    vertices={}
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
            # The graph goes as follows:
            #   \v[nv].p    /| v[k]
            #    \|       / 
            #   nv<--<--< k
            #    /      |\
            # |/  v[nv]   \ v[k].p
            # on the left side, the edge goes downward and on the right side, the edge goes upward (in the middle region)
            forward=HalfEdge(k)
            backward=HalfEdge(nv)
            # connect two edges
            forward.twin_half_edge=backward
            backward.twin_half_edge=forward
            # test dictionary value based edit
            # resolve backward edge
            if nv in vertices:
                nv_hi=half_edges[vertices[nv]].prev_half_edge
                logger.debug(f"link backward edge to previous half edge {nv_hi}")
                backward.prev_half_edge=nv_hi
                assert nv_hi.destination_vertex()==nv, f"nv_hi.destination_vertex()={vertices[nv].prev_half_edge.destination_vertex()}, nv={nv}"
                nv_hi.next_half_edge=backward
            else:
                backward.prev_half_edge=forward
            if k in vertices:
                k_hi=half_edges[vertices[k]]
                logger.debug(f"link backward edge to next half edge {k_hi}")
                backward.next_half_edge=k_hi
                assert k_hi.origin_vertex==k, f"k_hi.origin_vertex={vertices[k].origin_vertex}, k={k}"
                k_hi.prev_half_edge=backward
            else:
                backward.next_half_edge=forward
            # resolve forward edge
            if nv in vertices:
                nv_lo=half_edges[vertices[nv]]
                logger.debug(f"link forward edge to next half edge {nv_lo}")
                forward.next_half_edge=nv_lo
                assert nv_lo.origin_vertex==nv, f"nv_lo.origin_vertex={vertices[nv].origin_vertex}, nv={nv}"
                nv_lo.prev_half_edge=forward
            else:
                forward.next_half_edge=backward
            if k in vertices:
                k_lo=half_edges[vertices[k]].prev_half_edge
                logger.debug(f"link forward edge to previous half edge {k_lo}")
                forward.prev_half_edge=k_lo
                assert k_lo.destination_vertex()==k, f"k_lo.destination_vertex={vertices[k].prev_half_edge.destination_vertex()}, k={k}"    
                k_lo.next_half_edge=forward
            else:
                forward.prev_half_edge=backward
            # record half-edges
            half_edges.append(forward)
            half_edges.append(backward)
            vertices[k]=len(half_edges)-2
            vertices[nv]=len(half_edges)-1
            logger.debug(f"vertex {k} is connected to {nv}, forward half-edge: {forward}, backward half-edge: {backward}, updated vertices: {[(k,str(v)) for k,v in vertices.items()]}")
    # restore vertices address
    vertices_edges=dict()
    for k,v in vertices.items():
        vertices_edges[k]=half_edges[v]
    return vertices_edges,half_edges

def half_edges_to_segments(half_edges:list[HalfEdge]) -> list:
    """
    Convert half-edges to line segments.
    Arguments:
        half_edges: a list of half-edges
    Returns:
        segments: a list of line segments, no duplicates
    """
    segments=set()
    for half_edge in half_edges:
        s,e=half_edge.origin_vertex,half_edge.destination_vertex()
        if (s,e) in segments or (e,s) in segments:
            continue
        segments.add((s,e))
    return list(segments)

def assign_faces(half_edges:list[HalfEdge],initial_face_id:int=0) -> dict[int,HalfEdge]:
    """
    Assign faces to each half-edges
    Arguments:
        half_edges: a list of half-edges
        vertices: a dict of vertices
    Returns:
        faces: a list of faces with corresponding initial half-edges
    """
    # start with lower left vertex, default 0 is the infinite face
    face_id=initial_face_id
    faces=collections.defaultdict(HalfEdge)
    assert type(half_edges) is list and type(half_edges[0]) is HalfEdge, f"incorrect target type of half_edges: {type(half_edges)}"
    half_edges=sorted(half_edges,key=lambda x:x.origin_vertex)
    for half_edge in half_edges:
        if half_edge.face is None:
            # start traversal, expected to have loop for each
            origin=half_edge.origin_vertex
            faces[face_id]=half_edge
            half_edge=half_edge.next_half_edge
            while half_edge.origin_vertex!=origin:
                # move along the origin vertex
                assert not half_edge.face, f"half edge {half_edge} is already assigned with face {half_edge.face}, check your traversal order to debug."
                logger.debug(f"traversal half edge {half_edge}, face {face_id}")
                half_edge.face=face_id
                half_edge=half_edge.next_half_edge
            face_id+=1
    return faces,half_edges

def parking_adapter(p0s,p1s,p2s,x0s,x1s,x2s)->bool:
    """
    Return False iff all rectangles identified by p0s,p1s,p2s are contained in x0s,x1s,x2s
    Arguments:
        p0s,p1s,p2s: a list of 3 points, each point is a tuple of (x,y)
        x0s,x1s,x2s: a list of 3 points, each point is a tuple of (x,y)
    Returns:
        True iff all rectangles identified by p0s,p1s,p2s are contained in x0s,x1s,x2s
    """
    p3s=p1s+p2s-p0s
    x3s=x1s+x2s-x0s
    def nptotuple(npt):
        return tuple([float(x) for x in npt])
    print(p0s,p1s,p2s,p3s,x0s,x1s,x2s,x3s)
    segments1=[(nptotuple(x[0][0]),nptotuple(x[1][0])) for x in zip(p0s,p1s,p2s,p3s)]
    segments2=[(nptotuple(x[0]),nptotuple(x[1])) for x in zip(x0s,x1s,x2s,x3s)]
    print(segments1,segments2)
    logger.debug(f"segments1: {segments1}, segments2: {segments2}")
    return not map_overlays(segments1,segments2)

def map_overlays(segments1,segments2)->bool:
    """
    Check if map 1 created by segments1 is contained in map 2 created by segments2
    Arguments:
        segments1: a list of line segments, each segment is a tuple of ((x1,y1),(x2,y2))
        segments2: a list of line segments, each segment is a tuple of ((x1,y1),(x2,y2))
    Returns:
        True if map 1 is contained in map 2
    """
    v1,e1=get_planner_subdivision(segments1)
    v2,e2=get_planner_subdivision(segments2)
    # test merge map
    segments3=half_edges_to_segments(e1)+half_edges_to_segments(e2)
    v3,e3=get_planner_subdivision(segments3)
    # if more vertex is created, then there must be intersections
    if len(v3)>len(v1)+len(v2):
        return False
    # test face assignment for each graph and see if our interested face is a hole or not.
    # Hope you will not use it, it is hard to implement and need to rewrite the line intersection algorithm
    # logger.warning("map_overlays is not fully implemented.")
    # if the face in e1 is not 
    return True
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
        
    # special non-trivial test case for line intersection
    segments = [((3.0,1.0),(5.0,5.0)),
                ((4.0,0.0),(4.0,5.0)),
                ((6.0,1.0),(4.0,3.0)),
                ((7.0,1.0),(8.0,4.0)),
                ((2.0,3.0),(1.0,5.0)),
                ((4.0,3.0),(3.0,5.0)),
                ((4.0,3.0),(0.0,5.0))]
    
    # edge direction test case, expected half-edges to be sorted counter-clockwise with respect to the -x axis
    segments = [((2.0,2.0),(2.0,1.0)),
                ((1.0,1.0),(0.0,2.0)),
                ((1.0,2.0),(1.0,1.0)),
                ((2.0,2.0),(1.0,1.0)),
                ((1.0,1.0),(1.0,0.0)),
                ((0.0,1.0),(1.0,1.0)),
                ((1.0,1.0),(2.0,0.0))]
    
    # special non-trivial test case for polygon intersection
    segments1 = [((0.0,0.0),(3.0,0.0)),
                 ((3.0,0.0),(2.0,2.0)),
                 ((2.0,2.0),(3.0,3.0)),
                 ((3.0,3.0),(0.0,3.0)),
                 ((0.0,3.0),(0.0,0.0)),]
    segments2 = [((5.0,3.0),(5.0,0.0)),
                 ((5.0,0.0),(2.0,0.0)),
                 ((2.0,0.0),(3.0,1.0)),
                 ((3.0,1.0),(2.0,3.0)),
                 ((2.0,3.0),(5.0,3.0)),]
    
    # special non-trivial test case for no intersection but inside the polygon
    segments1 = [((0.0,0.0),(3.0,0.0)),
                 ((3.0,0.0),(3.0,3.0)),
                 ((3.0,3.0),(0.0,3.0)),
                 ((0.0,3.0),(0.0,0.0)),]
    segments2 = [((1.0,1.0),(2.0,1.0)),
                 ((2.0,1.0),(2.0,2.0)),
                 ((2.0,2.0),(1.0,2.0)),
                 ((1.0,2.0),(1.0,1.0)),]
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
    # vertices,half_edges = get_planner_subdivision(segments)
    # plt.scatter([x for x,y in vertices], [y for x,y in vertices.keys()],color='red')
    # for half_edges in half_edges:
    #     plt.plot([half_edges.origin_vertex[0],half_edges.destination_vertex()[0]], [half_edges.origin_vertex[1],half_edges.destination_vertex()[1]],color='blue')
    # plt.show()
    #########################################################
    # function test for 'planner subdivision' direction of edges
    # vertices,half_edges = get_planner_subdivision(segments)
    # plt.scatter([x for x,y in vertices], [y for x,y in vertices.keys()],color='red')
    # logger.info(f"resolved half-edges: {'\n'.join([str(half_edge) for half_edge in half_edges])}")
    # for half_edges in half_edges:
    #     plt.plot([half_edges.origin_vertex[0],half_edges.destination_vertex()[0]], [half_edges.origin_vertex[1],half_edges.destination_vertex()[1]],color='blue')
    # plt.show()
    #########################################################
    # function test for face assignment,
    vertices,half_edges = get_planner_subdivision(segments1)
    logger.info(f"resolved half-edges: {''.join([str(half_edge) for half_edge in half_edges])}")
    plt.scatter([x for x,y in vertices], [y for x,y in vertices.keys()],color='red')
    for half_edge in half_edges:
        plt.plot([half_edge.origin_vertex[0],half_edge.destination_vertex()[0]], [half_edge.origin_vertex[1],half_edge.destination_vertex()[1]],color='blue')
    vertices,half_edges = get_planner_subdivision(segments2)
    logger.info(f"resolved half-edges: {''.join([str(half_edge) for half_edge in half_edges])}")
    plt.scatter([x for x,y in vertices], [y for x,y in vertices.keys()],color='red')
    for half_edge in half_edges:
        plt.plot([half_edge.origin_vertex[0],half_edge.destination_vertex()[0]], [half_edge.origin_vertex[1],half_edge.destination_vertex()[1]],color='blue')
    
    faces,edges=assign_faces(half_edges)
    logger.info(f"assigned faces: {''.join([str(face)+': '+val.str() for face,val in faces.items()])}")
    plt.show()
    #########################################################