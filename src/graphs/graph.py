from __future__ import annotations

import heapq
from typing import Optional
from collections import deque


class GraphAdjMatrix:
    """
    Graph implementation using Adjacency Matrix.

    The adjacency matrix is a 2D array where matrix[i][j] represents
    the edge weight from vertex i to vertex j. For unweighted graphs,
    1 indicates an edge exists, 0 indicates no edge.

    Time Complexity:
        - add_vertex: O(V) - need to expand matrix
        - remove_vertex: O(V²) - need to shift rows/columns
        - add_edge: O(1)
        - remove_edge: O(1)
        - has_edge: O(1)
        - get_neighbors: O(V)
        - BFS/DFS: O(V²)

    Space Complexity: O(V²) - stores V×V matrix regardless of edges

    Best for: Dense graphs where E ≈ V²
    """

    def __init__(self, directed: bool = False) -> None:
        self.directed: bool = directed
        self.matrix: list[list[float]] = []
        self.vertex_map: dict[str, int] = {}  # vertex label -> index
        self.index_map: dict[int, str] = {}  # index -> vertex label
        self.vertex_count: int = 0

    def add_vertex(self, vertex: str) -> bool:
        """Add a vertex to the graph. O(V)"""
        if vertex in self.vertex_map:
            return False

        # Add new row and column to matrix
        for row in self.matrix:
            row.append(0)
        self.matrix.append([0] * (self.vertex_count + 1))

        # Update mappings
        self.vertex_map[vertex] = self.vertex_count
        self.index_map[self.vertex_count] = vertex
        self.vertex_count += 1
        return True

    def remove_vertex(self, vertex: str) -> bool:
        """Remove a vertex and all its edges. O(V²)"""
        if vertex not in self.vertex_map:
            return False

        index = self.vertex_map[vertex]

        # Remove row and column
        self.matrix.pop(index)
        for row in self.matrix:
            row.pop(index)

        # Update mappings
        del self.vertex_map[vertex]
        del self.index_map[index]

        # Shift indices for vertices after removed one
        new_vertex_map: dict[str, int] = {}
        new_index_map: dict[int, str] = {}
        for v, i in self.vertex_map.items():
            new_i = i if i < index else i - 1
            new_vertex_map[v] = new_i
            new_index_map[new_i] = v

        self.vertex_map = new_vertex_map
        self.index_map = new_index_map
        self.vertex_count -= 1
        return True

    def add_edge(self, from_vertex: str, to_vertex: str, weight: float = 1) -> bool:
        """Add an edge between two vertices. O(1)"""
        if from_vertex not in self.vertex_map or to_vertex not in self.vertex_map:
            return False

        from_idx = self.vertex_map[from_vertex]
        to_idx = self.vertex_map[to_vertex]

        self.matrix[from_idx][to_idx] = weight
        if not self.directed:
            self.matrix[to_idx][from_idx] = weight
        return True

    def remove_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """Remove an edge between two vertices. O(1)"""
        if from_vertex not in self.vertex_map or to_vertex not in self.vertex_map:
            return False

        from_idx = self.vertex_map[from_vertex]
        to_idx = self.vertex_map[to_vertex]

        if self.matrix[from_idx][to_idx] == 0:
            return False

        self.matrix[from_idx][to_idx] = 0
        if not self.directed:
            self.matrix[to_idx][from_idx] = 0
        return True

    def has_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """Check if an edge exists between two vertices. O(1)"""
        if from_vertex not in self.vertex_map or to_vertex not in self.vertex_map:
            return False

        from_idx = self.vertex_map[from_vertex]
        to_idx = self.vertex_map[to_vertex]
        return self.matrix[from_idx][to_idx] != 0

    def get_edge_weight(self, from_vertex: str, to_vertex: str) -> Optional[float]:
        """Get the weight of an edge. O(1)"""
        if not self.has_edge(from_vertex, to_vertex):
            return None

        from_idx = self.vertex_map[from_vertex]
        to_idx = self.vertex_map[to_vertex]
        return self.matrix[from_idx][to_idx]

    def get_neighbors(self, vertex: str) -> list[str]:
        """Get all neighbors of a vertex. O(V)"""
        if vertex not in self.vertex_map:
            return []

        idx = self.vertex_map[vertex]
        neighbors = []
        for i, weight in enumerate(self.matrix[idx]):
            if weight != 0:
                neighbors.append(self.index_map[i])
        return neighbors

    def get_vertices(self) -> list[str]:
        """Get all vertices in the graph. O(V)"""
        return list(self.vertex_map.keys())

    def get_edge_count(self) -> int:
        """Get the number of edges. O(V²)"""
        count = 0
        for i in range(self.vertex_count):
            for j in range(self.vertex_count):
                if self.matrix[i][j] != 0:
                    count += 1
        if not self.directed:
            count //= 2
        return count

    def bfs(self, start: str) -> list[str]:
        """Breadth-First Search traversal. O(V²)"""
        if start not in self.vertex_map:
            return []

        visited: set[str] = set()
        result: list[str] = []
        queue: deque[str] = deque([start])
        visited.add(start)

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            for neighbor in self.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def dfs(self, start: str) -> list[str]:
        """Depth-First Search traversal (iterative). O(V²)"""
        if start not in self.vertex_map:
            return []

        visited: set[str] = set()
        result: list[str] = []
        stack: list[str] = [start]

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)

                # Reverse to match recursive DFS order (stack is LIFO)
                for neighbor in reversed(self.get_neighbors(vertex)):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return result

    def dfs_recursive(self, start: str) -> list[str]:
        """Depth-First Search traversal (recursive). O(V²)"""
        if start not in self.vertex_map:
            return []

        visited: set[str] = set()
        result: list[str] = []
        self._dfs_helper(start, visited, result)
        return result

    def _dfs_helper(
        self, vertex: str, visited: set[str], result: list[str]
    ) -> None:
        """Helper for recursive DFS."""
        visited.add(vertex)
        result.append(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                self._dfs_helper(neighbor, visited, result)

    def has_path(self, from_vertex: str, to_vertex: str) -> bool:
        """Check if a path exists between two vertices using BFS. O(V²)"""
        if from_vertex not in self.vertex_map or to_vertex not in self.vertex_map:
            return False

        visited: set[str] = set()
        queue: deque[str] = deque([from_vertex])
        visited.add(from_vertex)

        while queue:
            vertex = queue.popleft()
            if vertex == to_vertex:
                return True

            for neighbor in self.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def topological_sort(self) -> list[str]:
        """
        Topological sort using DFS. O(V²)
        Returns vertices in topological order.
        Only valid for directed acyclic graphs (DAGs).
        Returns empty list if cycle detected.
        """
        if not self.directed:
            return []  # Only works for directed graphs

        visited: set[str] = set()
        rec_stack: set[str] = set()  # For cycle detection
        result: list[str] = []

        for vertex in self.get_vertices():
            if vertex not in visited:
                if not self._topological_sort_dfs(vertex, visited, rec_stack, result):
                    return []  # Cycle detected

        return result[::-1]  # Reverse for correct order

    def _topological_sort_dfs(
        self,
        vertex: str,
        visited: set[str],
        rec_stack: set[str],
        result: list[str],
    ) -> bool:
        """Helper for topological sort. Returns False if cycle detected."""
        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                if not self._topological_sort_dfs(neighbor, visited, rec_stack, result):
                    return False  # Propagate cycle detection upward
            elif neighbor in rec_stack:
                return False  # Cycle detected

        rec_stack.remove(vertex)
        result.append(vertex)  # Add after processing all neighbors
        return True

    def has_cycle(self) -> bool:
        """
        Check if the graph has a cycle. O(V²)
        Uses recursion stack for directed graphs.
        Uses parent tracking for undirected graphs.
        """
        visited: set[str] = set()

        if self.directed:
            rec_stack: set[str] = set()
            for vertex in self.get_vertices():
                if vertex not in visited:
                    if self._has_cycle_directed_dfs(vertex, visited, rec_stack):
                        return True
        else:
            for vertex in self.get_vertices():
                if vertex not in visited:
                    if self._has_cycle_undirected_dfs(vertex, visited, None):
                        return True
        return False

    def _has_cycle_directed_dfs(
        self,
        vertex: str,
        visited: set[str],
        rec_stack: set[str],
    ) -> bool:
        """Helper for directed cycle detection. Returns True if cycle detected."""
        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                if self._has_cycle_directed_dfs(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True  # Back edge = cycle

        rec_stack.remove(vertex)
        return False

    def _has_cycle_undirected_dfs(
        self,
        vertex: str,
        visited: set[str],
        parent: Optional[str],
    ) -> bool:
        """Helper for undirected cycle detection. Returns True if cycle detected."""
        visited.add(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                if self._has_cycle_undirected_dfs(neighbor, visited, vertex):
                    return True
            elif neighbor != parent:
                return True  # Visited neighbor that's not parent = cycle

        return False

    def shortest_path(
        self, from_vertex: str, to_vertex: str
    ) -> tuple[Optional[float], list[str]]:
        """
        Find shortest path using Dijkstra's algorithm. O(V² log V)
        Returns (distance, path) where path is list of vertices.
        Returns (None, []) if no path exists.
        Only works with non-negative edge weights.
        """
        if from_vertex not in self.vertex_map or to_vertex not in self.vertex_map:
            return None, []

        # Distance from source to each vertex
        distances: dict[str, float] = {v: float("inf") for v in self.vertex_map}
        distances[from_vertex] = 0

        # Track previous vertex in shortest path
        previous: dict[str, Optional[str]] = {v: None for v in self.vertex_map}

        # Priority queue: (distance, vertex)
        pq: list[tuple[float, str]] = [(0, from_vertex)]

        visited: set[str] = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            # Found destination
            if current == to_vertex:
                break

            # Check all neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                weight = self.get_edge_weight(current, neighbor) or 1
                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        # No path found
        if distances[to_vertex] == float("inf"):
            return None, []

        # Reconstruct path
        path: list[str] = []
        current = to_vertex
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return distances[to_vertex], path

    def shortest_distance(self, from_vertex: str, to_vertex: str) -> Optional[float]:
        """
        Find shortest distance using Dijkstra's algorithm. O(V² log V)
        Returns None if no path exists.
        """
        distance, _ = self.shortest_path(from_vertex, to_vertex)
        return distance

    def minimum_spanning_tree(
        self, start_vertex: Optional[str] = None
    ) -> tuple[float, list[tuple[str, str, float]]]:
        """
        Find Minimum Spanning Tree using Prim's algorithm. O(V² log V)
        Returns (total_weight, edges) where edges is list of (from, to, weight).
        Returns (0, []) for empty graph or directed graph.
        Only works for undirected, connected graphs.
        """
        if self.directed or self.vertex_count == 0:
            return 0, []

        # Start from given vertex or first vertex
        if start_vertex is None or start_vertex not in self.vertex_map:
            start_vertex = self.index_map[0]

        # Track visited vertices and MST edges
        visited: set[str] = set()
        mst_edges: list[tuple[str, str, float]] = []
        total_weight: float = 0

        # Priority queue: (weight, from_vertex, to_vertex)
        pq: list[tuple[float, str, str]] = [(0, start_vertex, start_vertex)]

        while pq and len(visited) < self.vertex_count:
            weight, from_v, to_v = heapq.heappop(pq)

            if to_v in visited:
                continue
            visited.add(to_v)

            # Add edge to MST (skip the starting vertex's dummy edge)
            if from_v != to_v:
                mst_edges.append((from_v, to_v, weight))
                total_weight += weight

            # Add all edges from current vertex to unvisited neighbors
            for neighbor in self.get_neighbors(to_v):
                if neighbor not in visited:
                    edge_weight = self.get_edge_weight(to_v, neighbor) or 1
                    heapq.heappush(pq, (edge_weight, to_v, neighbor))

        # Check if all vertices are included (graph is connected)
        if len(visited) != self.vertex_count:
            return 0, []  # Graph is not connected

        return total_weight, mst_edges

    def is_empty(self) -> bool:
        """Check if the graph is empty. O(1)"""
        return self.vertex_count == 0

    def __len__(self) -> int:
        """Return the number of vertices. O(1)"""
        return self.vertex_count

    def __str__(self) -> str:
        """String representation of the adjacency matrix."""
        if self.vertex_count == 0:
            return "Empty graph"

        # Header
        vertices = [self.index_map[i] for i in range(self.vertex_count)]
        header = "    " + " ".join(f"{v:>3}" for v in vertices)
        lines = [header]

        # Rows
        for i in range(self.vertex_count):
            row_label = f"{vertices[i]:>3}"
            row_values = " ".join(f"{int(self.matrix[i][j]):>3}" for j in range(self.vertex_count))
            lines.append(f"{row_label} {row_values}")

        return "\n".join(lines)


class GraphAdjList:
    """
    Graph implementation using Adjacency List.

    Each vertex stores a list of its neighbors. For weighted graphs,
    each neighbor is stored as (vertex, weight) tuple.

    Time Complexity:
        - add_vertex: O(1)
        - remove_vertex: O(V + E)
        - add_edge: O(1)
        - remove_edge: O(E) - need to find edge in list
        - has_edge: O(degree) - check neighbor list
        - get_neighbors: O(1)
        - BFS/DFS: O(V + E)

    Space Complexity: O(V + E) - stores only existing edges

    Best for: Sparse graphs where E << V²
    """

    def __init__(self, directed: bool = False) -> None:
        self.directed: bool = directed
        self.adj_list: dict[str, list[tuple[str, float]]] = {}

    def add_vertex(self, vertex: str) -> bool:
        """Add a vertex to the graph. O(1)"""
        if vertex in self.adj_list:
            return False
        self.adj_list[vertex] = []
        return True

    def remove_vertex(self, vertex: str) -> bool:
        """Remove a vertex and all its edges. O(V + E)"""
        if vertex not in self.adj_list:
            return False

        # Remove all edges TO this vertex from other vertices
        for v, edges in self.adj_list.items():
            self.adj_list[v] = [
                (neighbor, weight)
                for neighbor, weight in edges
                if neighbor != vertex
            ]

        # Remove the vertex itself
        del self.adj_list[vertex]
        return True

    def add_edge(self, from_vertex: str, to_vertex: str, weight: float = 1) -> bool:
        """Add an edge between two vertices. O(1)"""
        # Auto-create vertices if they don't exist
        if from_vertex not in self.adj_list:
            self.adj_list[from_vertex] = []
        if to_vertex not in self.adj_list:
            self.adj_list[to_vertex] = []

        # Check if edge already exists
        for neighbor, _ in self.adj_list[from_vertex]:
            if neighbor == to_vertex:
                return False

        self.adj_list[from_vertex].append((to_vertex, weight))
        if not self.directed:
            self.adj_list[to_vertex].append((from_vertex, weight))
        return True

    def remove_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """Remove an edge between two vertices. O(E)"""
        if from_vertex not in self.adj_list or to_vertex not in self.adj_list:
            return False

        # Find and remove edge
        original_len = len(self.adj_list[from_vertex])
        self.adj_list[from_vertex] = [
            (neighbor, weight)
            for neighbor, weight in self.adj_list[from_vertex]
            if neighbor != to_vertex
        ]

        if len(self.adj_list[from_vertex]) == original_len:
            return False  # Edge didn't exist

        if not self.directed:
            self.adj_list[to_vertex] = [
                (neighbor, weight)
                for neighbor, weight in self.adj_list[to_vertex]
                if neighbor != from_vertex
            ]
        return True

    def has_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """Check if an edge exists between two vertices. O(degree)"""
        if from_vertex not in self.adj_list:
            return False

        for neighbor, _ in self.adj_list[from_vertex]:
            if neighbor == to_vertex:
                return True
        return False

    def get_edge_weight(self, from_vertex: str, to_vertex: str) -> Optional[float]:
        """Get the weight of an edge. O(degree)"""
        if from_vertex not in self.adj_list:
            return None

        for neighbor, weight in self.adj_list[from_vertex]:
            if neighbor == to_vertex:
                return weight
        return None

    def get_neighbors(self, vertex: str) -> list[str]:
        """Get all neighbors of a vertex. O(1)"""
        if vertex not in self.adj_list:
            return []
        return [neighbor for neighbor, _ in self.adj_list[vertex]]

    def get_neighbors_with_weights(self, vertex: str) -> list[tuple[str, float]]:
        """Get all neighbors with edge weights. O(1)"""
        if vertex not in self.adj_list:
            return []
        return self.adj_list[vertex].copy()

    def get_vertices(self) -> list[str]:
        """Get all vertices in the graph. O(V)"""
        return list(self.adj_list.keys())

    def get_edge_count(self) -> int:
        """Get the number of edges. O(V)"""
        count = sum(len(neighbors) for neighbors in self.adj_list.values())
        if not self.directed:
            count //= 2
        return count

    def get_degree(self, vertex: str) -> int:
        """Get the degree of a vertex. O(1)"""
        if vertex not in self.adj_list:
            return 0
        return len(self.adj_list[vertex])

    def bfs(self, start: str) -> list[str]:
        """Breadth-First Search traversal. O(V + E)"""
        if start not in self.adj_list:
            return []

        visited: set[str] = set()
        result: list[str] = []
        queue: deque[str] = deque([start])
        visited.add(start)

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            for neighbor in self.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def dfs(self, start: str) -> list[str]:
        """Depth-First Search traversal (iterative). O(V + E)"""
        if start not in self.adj_list:
            return []

        visited: set[str] = set()
        result: list[str] = []
        stack: list[str] = [start]

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)

                # Reverse to match recursive DFS order (stack is LIFO)
                for neighbor in reversed(self.get_neighbors(vertex)):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return result

    def dfs_recursive(self, start: str) -> list[str]:
        """Depth-First Search traversal (recursive). O(V + E)"""
        if start not in self.adj_list:
            return []

        visited: set[str] = set()
        result: list[str] = []
        self._dfs_helper(start, visited, result)
        return result

    def _dfs_helper(
        self, vertex: str, visited: set[str], result: list[str]
    ) -> None:
        """Helper for recursive DFS."""
        visited.add(vertex)
        result.append(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                self._dfs_helper(neighbor, visited, result)

    def has_path(self, from_vertex: str, to_vertex: str) -> bool:
        """Check if a path exists between two vertices using BFS. O(V + E)"""
        if from_vertex not in self.adj_list or to_vertex not in self.adj_list:
            return False

        visited: set[str] = set()
        queue: deque[str] = deque([from_vertex])
        visited.add(from_vertex)

        while queue:
            vertex = queue.popleft()
            if vertex == to_vertex:
                return True

            for neighbor in self.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def topological_sort(self) -> list[str]:
        """
        Topological sort using DFS. O(V + E)
        Returns vertices in topological order.
        Only valid for directed acyclic graphs (DAGs).
        Returns empty list if cycle detected.
        """
        if not self.directed:
            return []  # Only works for directed graphs

        visited: set[str] = set()
        rec_stack: set[str] = set()  # For cycle detection
        result: list[str] = []

        for vertex in self.get_vertices():
            if vertex not in visited:
                if not self._topological_sort_dfs(vertex, visited, rec_stack, result):
                    return []  # Cycle detected

        return result[::-1]  # Reverse for correct order

    def _topological_sort_dfs(
        self,
        vertex: str,
        visited: set[str],
        rec_stack: set[str],
        result: list[str],
    ) -> bool:
        """Helper for topological sort. Returns False if cycle detected."""
        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                if not self._topological_sort_dfs(neighbor, visited, rec_stack, result):
                    return False  # Propagate cycle detection upward
            elif neighbor in rec_stack:
                return False  # Cycle detected

        rec_stack.remove(vertex)
        result.append(vertex)  # Add after processing all neighbors
        return True

    def has_cycle(self) -> bool:
        """
        Check if the graph has a cycle. O(V + E)
        Uses recursion stack for directed graphs.
        Uses parent tracking for undirected graphs.
        """
        visited: set[str] = set()

        if self.directed:
            rec_stack: set[str] = set()
            for vertex in self.get_vertices():
                if vertex not in visited:
                    if self._has_cycle_directed_dfs(vertex, visited, rec_stack):
                        return True
        else:
            for vertex in self.get_vertices():
                if vertex not in visited:
                    if self._has_cycle_undirected_dfs(vertex, visited, None):
                        return True
        return False

    def _has_cycle_directed_dfs(
        self,
        vertex: str,
        visited: set[str],
        rec_stack: set[str],
    ) -> bool:
        """Helper for directed cycle detection. Returns True if cycle detected."""
        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                if self._has_cycle_directed_dfs(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True  # Back edge = cycle

        rec_stack.remove(vertex)
        return False

    def _has_cycle_undirected_dfs(
        self,
        vertex: str,
        visited: set[str],
        parent: Optional[str],
    ) -> bool:
        """Helper for undirected cycle detection. Returns True if cycle detected."""
        visited.add(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                if self._has_cycle_undirected_dfs(neighbor, visited, vertex):
                    return True
            elif neighbor != parent:
                return True  # Visited neighbor that's not parent = cycle

        return False

    def shortest_path(
        self, from_vertex: str, to_vertex: str
    ) -> tuple[Optional[float], list[str]]:
        """
        Find shortest path using Dijkstra's algorithm. O((V + E) log V)
        Returns (distance, path) where path is list of vertices.
        Returns (None, []) if no path exists.
        Only works with non-negative edge weights.
        """
        if from_vertex not in self.adj_list or to_vertex not in self.adj_list:
            return None, []

        # Distance from source to each vertex
        distances: dict[str, float] = {v: float("inf") for v in self.adj_list}
        distances[from_vertex] = 0

        # Track previous vertex in shortest path
        previous: dict[str, Optional[str]] = {v: None for v in self.adj_list}

        # Priority queue: (distance, vertex)
        pq: list[tuple[float, str]] = [(0, from_vertex)]

        visited: set[str] = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            # Found destination
            if current == to_vertex:
                break

            # Check all neighbors with weights
            for neighbor, weight in self.adj_list[current]:
                if neighbor in visited:
                    continue

                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        # No path found
        if distances[to_vertex] == float("inf"):
            return None, []

        # Reconstruct path
        path: list[str] = []
        current = to_vertex
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return distances[to_vertex], path

    def shortest_distance(self, from_vertex: str, to_vertex: str) -> Optional[float]:
        """
        Find shortest distance using Dijkstra's algorithm. O((V + E) log V)
        Returns None if no path exists.
        """
        distance, _ = self.shortest_path(from_vertex, to_vertex)
        return distance

    def minimum_spanning_tree(
        self, start_vertex: Optional[str] = None
    ) -> tuple[float, list[tuple[str, str, float]]]:
        """
        Find Minimum Spanning Tree using Prim's algorithm. O((V + E) log V)
        Returns (total_weight, edges) where edges is list of (from, to, weight).
        Returns (0, []) for empty graph or directed graph.
        Only works for undirected, connected graphs.
        """
        if self.directed or not self.adj_list:
            return 0, []

        # Start from given vertex or first vertex
        if start_vertex is None or start_vertex not in self.adj_list:
            start_vertex = next(iter(self.adj_list))

        # Track visited vertices and MST edges
        visited: set[str] = set()
        mst_edges: list[tuple[str, str, float]] = []
        total_weight: float = 0

        # Priority queue: (weight, from_vertex, to_vertex)
        pq: list[tuple[float, str, str]] = [(0, start_vertex, start_vertex)]

        while pq and len(visited) < len(self.adj_list):
            weight, from_v, to_v = heapq.heappop(pq)

            if to_v in visited:
                continue
            visited.add(to_v)

            # Add edge to MST (skip the starting vertex's dummy edge)
            if from_v != to_v:
                mst_edges.append((from_v, to_v, weight))
                total_weight += weight

            # Add all edges from current vertex to unvisited neighbors
            for neighbor, edge_weight in self.adj_list[to_v]:
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_weight, to_v, neighbor))

        # Check if all vertices are included (graph is connected)
        if len(visited) != len(self.adj_list):
            return 0, []  # Graph is not connected

        return total_weight, mst_edges

    def is_empty(self) -> bool:
        """Check if the graph is empty. O(1)"""
        return len(self.adj_list) == 0

    def __len__(self) -> int:
        """Return the number of vertices. O(1)"""
        return len(self.adj_list)

    def __str__(self) -> str:
        """String representation of the adjacency list."""
        if not self.adj_list:
            return "Empty graph"

        lines = []
        for vertex in sorted(self.adj_list.keys()):
            neighbors = self.adj_list[vertex]
            if neighbors:
                neighbor_str = ", ".join(
                    f"{n}({w})" if w != 1 else n for n, w in neighbors
                )
                lines.append(f"{vertex}: {neighbor_str}")
            else:
                lines.append(f"{vertex}: (no edges)")
        return "\n".join(lines)


def test_graph_adj_matrix() -> None:
    """Test GraphAdjMatrix operations."""
    print("=" * 60)
    print("Testing GraphAdjMatrix (Adjacency Matrix)")
    print("=" * 60)

    # Test undirected graph
    print("\n1. Creating undirected graph:")
    graph = GraphAdjMatrix(directed=False)
    print(f"   is_empty(): {graph.is_empty()}")

    # Add vertices
    print("\n2. Adding vertices:")
    for v in ["A", "B", "C", "D", "E"]:
        graph.add_vertex(v)
        print(f"   add_vertex('{v}'): vertex_count={len(graph)}")

    # Add edges
    print("\n3. Adding edges:")
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    for from_v, to_v in edges:
        graph.add_edge(from_v, to_v)
        print(f"   add_edge('{from_v}', '{to_v}')")

    # Display matrix
    print("\n4. Adjacency Matrix:")
    for line in str(graph).split("\n"):
        print(f"   {line}")

    # Test has_edge
    print("\n5. Testing has_edge():")
    test_edges = [("A", "B"), ("A", "D"), ("B", "A"), ("E", "A")]
    for from_v, to_v in test_edges:
        print(f"   has_edge('{from_v}', '{to_v}'): {graph.has_edge(from_v, to_v)}")

    # Test get_neighbors
    print("\n6. Testing get_neighbors():")
    for v in ["A", "D", "E"]:
        print(f"   get_neighbors('{v}'): {graph.get_neighbors(v)}")

    # Test BFS
    print("\n7. Testing BFS traversal:")
    print(f"   bfs('A'): {graph.bfs('A')}")

    # Test DFS
    print("\n8. Testing DFS traversal:")
    print(f"   dfs('A'): {graph.dfs('A')}")
    print(f"   dfs_recursive('A'): {graph.dfs_recursive('A')}")

    # Test has_path
    print("\n9. Testing has_path():")
    print(f"   has_path('A', 'E'): {graph.has_path('A', 'E')}")
    print(f"   has_path('E', 'A'): {graph.has_path('E', 'A')}")

    # Test remove_edge
    print("\n10. Testing remove_edge():")
    print(f"   remove_edge('D', 'E'): {graph.remove_edge('D', 'E')}")
    print(f"   has_path('A', 'E'): {graph.has_path('A', 'E')}")

    # Test directed graph
    print("\n11. Creating directed graph:")
    digraph = GraphAdjMatrix(directed=True)
    for v in ["A", "B", "C"]:
        digraph.add_vertex(v)
    digraph.add_edge("A", "B")
    digraph.add_edge("B", "C")
    print(f"   has_edge('A', 'B'): {digraph.has_edge('A', 'B')}")
    print(f"   has_edge('B', 'A'): {digraph.has_edge('B', 'A')}")

    # Test weighted graph
    print("\n12. Testing weighted graph:")
    wgraph = GraphAdjMatrix(directed=False)
    for v in ["A", "B", "C"]:
        wgraph.add_vertex(v)
    wgraph.add_edge("A", "B", 5)
    wgraph.add_edge("B", "C", 3)
    print(f"   get_edge_weight('A', 'B'): {wgraph.get_edge_weight('A', 'B')}")
    print(f"   get_edge_weight('B', 'C'): {wgraph.get_edge_weight('B', 'C')}")

    # Test topological sort
    print("\n13. Testing topological_sort():")
    dag = GraphAdjMatrix(directed=True)
    for v in ["A", "B", "C", "D", "E"]:
        dag.add_vertex(v)
    # A → B → D
    # A → C → D → E
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    dag.add_edge("D", "E")
    print(f"   DAG edges: A→B, A→C, B→D, C→D, D→E")
    print(f"   topological_sort(): {dag.topological_sort()}")

    # Test cycle detection with topological_sort
    print("\n14. Testing cycle detection with topological_sort:")
    cyclic = GraphAdjMatrix(directed=True)
    for v in ["A", "B", "C"]:
        cyclic.add_vertex(v)
    cyclic.add_edge("A", "B")
    cyclic.add_edge("B", "C")
    cyclic.add_edge("C", "A")  # Creates cycle
    print(f"   Cyclic graph: A→B→C→A")
    print(f"   topological_sort(): {cyclic.topological_sort()} (empty = cycle detected)")

    # Test has_cycle()
    print("\n15. Testing has_cycle():")
    print(f"   DAG (no cycle): has_cycle() = {dag.has_cycle()}")
    print(f"   Cyclic graph (A→B→C→A): has_cycle() = {cyclic.has_cycle()}")
    # Test undirected graph with cycle
    undirected_cyclic = GraphAdjMatrix(directed=False)
    for v in ["A", "B", "C"]:
        undirected_cyclic.add_vertex(v)
    undirected_cyclic.add_edge("A", "B")
    undirected_cyclic.add_edge("B", "C")
    undirected_cyclic.add_edge("C", "A")  # Creates cycle
    print(f"   Undirected cyclic (A-B-C-A): has_cycle() = {undirected_cyclic.has_cycle()}")
    # Test undirected graph without cycle (tree)
    undirected_tree = GraphAdjMatrix(directed=False)
    for v in ["A", "B", "C", "D"]:
        undirected_tree.add_vertex(v)
    undirected_tree.add_edge("A", "B")
    undirected_tree.add_edge("A", "C")
    undirected_tree.add_edge("C", "D")
    print(f"   Undirected tree (no cycle): has_cycle() = {undirected_tree.has_cycle()}")

    # Test shortest path
    print("\n16. Testing shortest_path() and shortest_distance():")
    # Create weighted graph for shortest path
    sp_graph = GraphAdjMatrix(directed=False)
    for v in ["A", "B", "C", "D", "E"]:
        sp_graph.add_vertex(v)
    sp_graph.add_edge("A", "B", 4)
    sp_graph.add_edge("A", "C", 2)
    sp_graph.add_edge("B", "C", 1)
    sp_graph.add_edge("B", "D", 5)
    sp_graph.add_edge("C", "D", 8)
    sp_graph.add_edge("C", "E", 10)
    sp_graph.add_edge("D", "E", 2)
    print("   Weighted graph:")
    print("       A --4-- B --5-- D")
    print("       |       |       |")
    print("       2       1       2")
    print("       |       |       |")
    print("       C --8-- + --10- E")
    dist, path = sp_graph.shortest_path("A", "E")
    print(f"   shortest_path('A', 'E'): distance={dist}, path={path}")
    print(f"   shortest_distance('A', 'D'): {sp_graph.shortest_distance('A', 'D')}")
    # Test no path
    sp_graph.add_vertex("F")  # Isolated vertex
    dist, path = sp_graph.shortest_path("A", "F")
    print(f"   shortest_path('A', 'F'): distance={dist}, path={path} (no path)")

    # Test minimum spanning tree
    print("\n17. Testing minimum_spanning_tree():")
    mst_graph = GraphAdjMatrix(directed=False)
    for v in ["A", "B", "C", "D"]:
        mst_graph.add_vertex(v)
    mst_graph.add_edge("A", "B", 4)
    mst_graph.add_edge("A", "C", 2)
    mst_graph.add_edge("B", "C", 1)
    mst_graph.add_edge("B", "D", 5)
    mst_graph.add_edge("C", "D", 3)
    print("   Weighted graph:")
    print("       A --4-- B")
    print("       |\\     /|")
    print("       2  \\1/  5")
    print("       |   X   |")
    print("       C --3-- D")
    total, edges = mst_graph.minimum_spanning_tree()
    print(f"   minimum_spanning_tree(): total_weight={total}, edges={edges}")
    # Test directed graph (should return empty)
    directed = GraphAdjMatrix(directed=True)
    for v in ["A", "B"]:
        directed.add_vertex(v)
    directed.add_edge("A", "B", 1)
    total, edges = directed.minimum_spanning_tree()
    print(f"   Directed graph MST: total_weight={total}, edges={edges} (not supported)")


def test_graph_adj_list() -> None:
    """Test GraphAdjList operations."""
    print("\n" + "=" * 60)
    print("Testing GraphAdjList (Adjacency List)")
    print("=" * 60)

    # Test undirected graph
    print("\n1. Creating undirected graph:")
    graph = GraphAdjList(directed=False)
    print(f"   is_empty(): {graph.is_empty()}")

    # Add vertices
    print("\n2. Adding vertices:")
    for v in ["A", "B", "C", "D", "E"]:
        graph.add_vertex(v)
        print(f"   add_vertex('{v}'): vertex_count={len(graph)}")

    # Add edges
    print("\n3. Adding edges:")
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    for from_v, to_v in edges:
        graph.add_edge(from_v, to_v)
        print(f"   add_edge('{from_v}', '{to_v}')")

    # Display adjacency list
    print("\n4. Adjacency List:")
    for line in str(graph).split("\n"):
        print(f"   {line}")

    # Test has_edge
    print("\n5. Testing has_edge():")
    test_edges = [("A", "B"), ("A", "D"), ("B", "A"), ("E", "A")]
    for from_v, to_v in test_edges:
        print(f"   has_edge('{from_v}', '{to_v}'): {graph.has_edge(from_v, to_v)}")

    # Test get_neighbors
    print("\n6. Testing get_neighbors():")
    for v in ["A", "D", "E"]:
        print(f"   get_neighbors('{v}'): {graph.get_neighbors(v)}")

    # Test get_degree
    print("\n7. Testing get_degree():")
    for v in ["A", "D", "E"]:
        print(f"   get_degree('{v}'): {graph.get_degree(v)}")

    # Test BFS
    print("\n8. Testing BFS traversal:")
    print(f"   bfs('A'): {graph.bfs('A')}")

    # Test DFS
    print("\n9. Testing DFS traversal:")
    print(f"   dfs('A'): {graph.dfs('A')}")
    print(f"   dfs_recursive('A'): {graph.dfs_recursive('A')}")

    # Test has_path
    print("\n10. Testing has_path():")
    print(f"   has_path('A', 'E'): {graph.has_path('A', 'E')}")
    print(f"   has_path('E', 'A'): {graph.has_path('E', 'A')}")

    # Test remove_edge
    print("\n11. Testing remove_edge():")
    print(f"   remove_edge('D', 'E'): {graph.remove_edge('D', 'E')}")
    print(f"   has_path('A', 'E'): {graph.has_path('A', 'E')}")

    # Test remove_vertex
    print("\n12. Testing remove_vertex():")
    print(f"   Before: vertices = {graph.get_vertices()}")
    print(f"   remove_vertex('C'): {graph.remove_vertex('C')}")
    print(f"   After: vertices = {graph.get_vertices()}")
    print(f"   get_neighbors('A'): {graph.get_neighbors('A')}")

    # Test directed graph
    print("\n13. Creating directed graph:")
    digraph = GraphAdjList(directed=True)
    digraph.add_edge("A", "B")
    digraph.add_edge("B", "C")
    print(f"   has_edge('A', 'B'): {digraph.has_edge('A', 'B')}")
    print(f"   has_edge('B', 'A'): {digraph.has_edge('B', 'A')}")

    # Test weighted graph
    print("\n14. Testing weighted graph:")
    wgraph = GraphAdjList(directed=False)
    wgraph.add_edge("A", "B", 5)
    wgraph.add_edge("B", "C", 3)
    wgraph.add_edge("A", "C", 10)
    print(f"   get_edge_weight('A', 'B'): {wgraph.get_edge_weight('A', 'B')}")
    print(f"   get_neighbors_with_weights('A'): {wgraph.get_neighbors_with_weights('A')}")
    print("\n   Weighted adjacency list:")
    for line in str(wgraph).split("\n"):
        print(f"   {line}")

    # Test topological sort
    print("\n15. Testing topological_sort():")
    dag = GraphAdjList(directed=True)
    # A → B → D
    # A → C → D → E
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    dag.add_edge("D", "E")
    print(f"   DAG edges: A→B, A→C, B→D, C→D, D→E")
    print(f"   topological_sort(): {dag.topological_sort()}")

    # Test cycle detection with topological_sort
    print("\n16. Testing cycle detection with topological_sort:")
    cyclic = GraphAdjList(directed=True)
    cyclic.add_edge("A", "B")
    cyclic.add_edge("B", "C")
    cyclic.add_edge("C", "A")  # Creates cycle
    print(f"   Cyclic graph: A→B→C→A")
    print(f"   topological_sort(): {cyclic.topological_sort()} (empty = cycle detected)")

    # Test has_cycle()
    print("\n17. Testing has_cycle():")
    print(f"   DAG (no cycle): has_cycle() = {dag.has_cycle()}")
    print(f"   Cyclic graph (A→B→C→A): has_cycle() = {cyclic.has_cycle()}")
    # Test undirected graph with cycle
    undirected_cyclic = GraphAdjList(directed=False)
    undirected_cyclic.add_edge("A", "B")
    undirected_cyclic.add_edge("B", "C")
    undirected_cyclic.add_edge("C", "A")  # Creates cycle
    print(f"   Undirected cyclic (A-B-C-A): has_cycle() = {undirected_cyclic.has_cycle()}")
    # Test undirected graph without cycle (tree)
    undirected_tree = GraphAdjList(directed=False)
    undirected_tree.add_edge("A", "B")
    undirected_tree.add_edge("A", "C")
    undirected_tree.add_edge("C", "D")
    print(f"   Undirected tree (no cycle): has_cycle() = {undirected_tree.has_cycle()}")

    # Test shortest path
    print("\n18. Testing shortest_path() and shortest_distance():")
    # Create weighted graph for shortest path
    sp_graph = GraphAdjList(directed=False)
    sp_graph.add_edge("A", "B", 4)
    sp_graph.add_edge("A", "C", 2)
    sp_graph.add_edge("B", "C", 1)
    sp_graph.add_edge("B", "D", 5)
    sp_graph.add_edge("C", "D", 8)
    sp_graph.add_edge("C", "E", 10)
    sp_graph.add_edge("D", "E", 2)
    print("   Weighted graph:")
    print("       A --4-- B --5-- D")
    print("       |       |       |")
    print("       2       1       2")
    print("       |       |       |")
    print("       C --8-- + --10- E")
    dist, path = sp_graph.shortest_path("A", "E")
    print(f"   shortest_path('A', 'E'): distance={dist}, path={path}")
    print(f"   shortest_distance('A', 'D'): {sp_graph.shortest_distance('A', 'D')}")
    # Test no path
    sp_graph.add_vertex("F")  # Isolated vertex
    dist, path = sp_graph.shortest_path("A", "F")
    print(f"   shortest_path('A', 'F'): distance={dist}, path={path} (no path)")

    # Test minimum spanning tree
    print("\n19. Testing minimum_spanning_tree():")
    mst_graph = GraphAdjList(directed=False)
    mst_graph.add_edge("A", "B", 4)
    mst_graph.add_edge("A", "C", 2)
    mst_graph.add_edge("B", "C", 1)
    mst_graph.add_edge("B", "D", 5)
    mst_graph.add_edge("C", "D", 3)
    print("   Weighted graph:")
    print("       A --4-- B")
    print("       |\\     /|")
    print("       2  \\1/  5")
    print("       |   X   |")
    print("       C --3-- D")
    total, edges = mst_graph.minimum_spanning_tree()
    print(f"   minimum_spanning_tree(): total_weight={total}, edges={edges}")
    # Test directed graph (should return empty)
    directed = GraphAdjList(directed=True)
    directed.add_edge("A", "B", 1)
    total, edges = directed.minimum_spanning_tree()
    print(f"   Directed graph MST: total_weight={total}, edges={edges} (not supported)")


def test_graph_comparison() -> None:
    """Compare both Graph implementations."""
    print("\n" + "=" * 60)
    print("Graph Implementations Comparison")
    print("=" * 60)

    # Create same graph with both implementations
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]

    matrix_graph = GraphAdjMatrix(directed=False)
    list_graph = GraphAdjList(directed=False)

    for v in ["A", "B", "C", "D", "E"]:
        matrix_graph.add_vertex(v)
        list_graph.add_vertex(v)

    for from_v, to_v in edges:
        matrix_graph.add_edge(from_v, to_v)
        list_graph.add_edge(from_v, to_v)

    print("\n1. Same graph, different representations:")
    print(f"\n   Adjacency Matrix:")
    for line in str(matrix_graph).split("\n"):
        print(f"   {line}")
    print(f"\n   Adjacency List:")
    for line in str(list_graph).split("\n"):
        print(f"   {line}")

    # Compare operations
    print("\n2. Operation results comparison:")
    print(f"   {'Operation':<30} {'Matrix':<15} {'List':<15}")
    print(f"   {'-' * 30} {'-' * 15} {'-' * 15}")
    print(f"   {'len(graph)':<30} {len(matrix_graph):<15} {len(list_graph):<15}")
    print(f"   {'get_edge_count()':<30} {matrix_graph.get_edge_count():<15} {list_graph.get_edge_count():<15}")
    print(f"   {'has_edge(A, B)':<30} {str(matrix_graph.has_edge('A', 'B')):<15} {str(list_graph.has_edge('A', 'B')):<15}")
    print(f"   {'has_edge(A, E)':<30} {str(matrix_graph.has_edge('A', 'E')):<15} {str(list_graph.has_edge('A', 'E')):<15}")
    print(f"   {'get_neighbors(D)':<30} {str(matrix_graph.get_neighbors('D')):<15} {str(list_graph.get_neighbors('D')):<15}")
    print(f"   {'bfs(A)':<30} {str(matrix_graph.bfs('A')):<15} {str(list_graph.bfs('A')):<15}")
    print(f"   {'has_path(A, E)':<30} {str(matrix_graph.has_path('A', 'E')):<15} {str(list_graph.has_path('A', 'E')):<15}")

    # Implementation comparison
    print("\n" + "=" * 60)
    print("Implementation Comparison")
    print("=" * 60)
    print(f"\n{'Aspect':<20} {'Adjacency Matrix':<25} {'Adjacency List':<25}")
    print(f"{'-' * 20} {'-' * 25} {'-' * 25}")
    print(f"{'Space':<20} {'O(V²)':<25} {'O(V + E)':<25}")
    print(f"{'Add vertex':<20} {'O(V)':<25} {'O(1)':<25}")
    print(f"{'Remove vertex':<20} {'O(V²)':<25} {'O(V + E)':<25}")
    print(f"{'Add edge':<20} {'O(1)':<25} {'O(1)':<25}")
    print(f"{'Remove edge':<20} {'O(1)':<25} {'O(E)':<25}")
    print(f"{'Has edge':<20} {'O(1)':<25} {'O(degree)':<25}")
    print(f"{'Get neighbors':<20} {'O(V)':<25} {'O(degree)':<25}")
    print(f"{'BFS/DFS':<20} {'O(V²)':<25} {'O(V + E)':<25}")

    print("\n" + "=" * 60)
    print("When to use which?")
    print("=" * 60)
    print("\nAdjacency Matrix:")
    print("   - Dense graphs (E ≈ V²)")
    print("   - Frequent edge existence checks")
    print("   - Need O(1) edge lookup")
    print("   - Small number of vertices")
    print("\nAdjacency List:")
    print("   - Sparse graphs (E << V²)")
    print("   - Memory constrained")
    print("   - Frequent neighbor iteration")
    print("   - Large graphs with few edges per vertex")
