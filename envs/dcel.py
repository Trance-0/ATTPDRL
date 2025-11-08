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
from heapq import heappush, heappop
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

class DCEL:
    def __init__(self, segments:list):
        """
        Initialize a DCEL.
        Arguments:
            segments: a list of line segments, each segment is a tuple of (x1,y1,x2,y2)
        """
        self.vertices,self.half_edges = self.__line_segment_intersection(segments)
        # faces is only constructed when the DCEL is fully constructed. takes O(n) time.
        # number of faces is (number of outer cycles + 1)
        # self.faces = []

    def __line_segment_intersection(self, segments:list) -> tuple(set,list):
        """
        Resolve intersections between line segments using plane sweep algorithm.
        Arguments:
            segments: a list of line segments, each segment is a tuple of (x1,y1,x2,y2)
        Returns:
            A resolved list of vertices and half-edges.
        """
        def intersect(segment1:tuple[float,float,float,float],segment2:tuple[float,float,float,float]) -> tuple[float,float] | None:
            """
            Compute the intersection of two line segments.
            Arguments:
                segment1: a tuple of (x1,y1,x2,y2)
                segment2: a tuple of (x3,y3,x4,y4)
            Returns:
                The intersection point of the two line segments.
                None if the two line segments are parallel.
            """
            x1,y1,x2,y2 = segment1
            x3,y3,x4,y4 = segment2
            denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            if denominator == 0:
                return None
            x = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denominator
            y = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denominator
            return x,y
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
        return set(),[]
        

    def add_vertex(self, vertex) -> bool:
        """
        Add a vertex to the DCEL if it is not already in the DCEL.
        Arguments:
            vertex: a tuple of (x,y) coordinates
        Returns:
            True if the vertex is added successfully, False if the vertex is already in the DCEL.
        """
        if vertex in self.vertices:
            return False
        self.vertices.add(vertex)
        return True
    
    def connect_vertices(self, vertex1, vertex2) -> bool:
        """
        Connect two vertices with two half-edges if they are in the DCEL.
        Arguments:
            vertex1: a tuple of (x,y) coordinates
            vertex2: a tuple of (x,y) coordinates
        Returns:
            True if the vertices are connected successfully, False if some vertices are not in the DCEL.
        """
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            return False
        half_edge1 = HalfEdge(vertex1)
        half_edge2 = HalfEdge(vertex2)
        return True