// A C++ program for Dijkstra's single source shortest path algorithm.
// The program is for adjacency matrix representation of the graph
#include <iostream>
#include <limits.h>

#include <CL/sycl.hpp>
using namespace cl::sycl;

// Number of vertices in the graph
#define V 9

// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(int dist[], bool sptSet[])
{

	// Initialize min value
	int min = INT_MAX, min_index;

	for (int v = 0; v < V; v++)
		if (sptSet[v] == false && dist[v] <= min)
			min = dist[v], min_index = v;

	return min_index;
}

// A utility function to print the constructed distance array
void printSolution(int dist[])
{
	std::cout << "Vertex \t Distance from Source" << std::endl;
	for (int i = 0; i < V; i++)
		std::cout << i << " \t\t" << dist[i] << std::endl;
}

// Function that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
//void dijkstra(int graph[V][V], int src)
void dijkstra(buffer<int, 2>& graph_buf, int src, queue& q)
{
	//int dist[V]; // The output array.  dist[i] will hold the shortest
	// distance from src to i
	buffer<int> dist_buf(V);

	//bool sptSet[V]; // sptSet[i] will be true if vertex i is included in shortest
	// path tree or shortest distance from src to i is finalized
	buffer<bool> sptSet_buf(V);

	// scoped host accessor
	{
		host_accessor dist{ dist_buf, write_only };
		host_accessor sptSet{ sptSet_buf, write_only };

		// Initialize all distances as INFINITE and stpSet[] as false
		for (int i = 0; i < V; i++)
			dist[i] = INT_MAX, sptSet[i] = false;

		// Distance of source vertex from itself is always 0
		dist[src] = 0;
	}
	
	std::pair<int, int> operator_identity = { 
		std::numeric_limits<int>::max(),
		std::numeric_limits<int>::min() 
	};
	buffer<decltype(operator_identity)> result_buf(1);

	// Find shortest path for all vertices
	for (int count = 0; count < V - 1; count++) {
		// Pick the minimum distance vertex from the set of vertices not
		// yet processed. u is always equal to src in the first iteration.
		int u;

		q.submit([&](handler& h) {
			accessor dist{ dist_buf, read_only };
			accessor sptSet{ sptSet_buf, read_only };

			auto reduction_object = reduction(
				result_buf,
				h,
				operator_identity,
				minimum<std::pair<int, int>>()
			);

			//# nd-range kernel parallel_for with reduction parameter
			h.parallel_for(range(V), reduction_object, [=](auto it, auto& temp) {
				auto v = it.get_global_id(0);
				if (sptSet[v] == false) {
					temp.combine({dist[v], v});
				}
			});
		});

		// scoped host accessor
		{
			host_accessor result{ result_buf, read_only };
			u = result[0].second;
		}

		// Mark the picked vertex as processed
		q.submit([&](handler& h) {
			accessor sptSet{ sptSet_buf, write_only };
			h.single_task([=]() {
				sptSet[u] = true;
			});
		});
		

		// Update dist value of the adjacent vertices of the picked vertex.
		for (int v = 0; v < V; v++)

			// Update dist[v] only if is not in sptSet, there is an edge from
			// u to v, and total weight of path from src to  v through u is
			// smaller than current value of dist[v]
			if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX
				&& dist[u] + graph[u][v] < dist[v])
				dist[v] = dist[u] + graph[u][v];

		// Update dist value of the adjacent vertices of the picked vertex.
		q.submit([&](handler& h) {
			accessor dist{ dist_buf, read_write };
			accessor sptSet{ sptSet_buf, read_only };
			accessor graph{ graph_buf, read_only };

			// Update dist[v] only if is not in sptSet, there is an edge from
			// u to v, and total weight of path from src to  v through u is
			// smaller than current value of dist[v]
			h.parallel_for(range(V), [=](auto it) {
				auto v = it.get_global_id(0);
				if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX
					&& dist[u] + graph[u][v] < dist[v])
					dist[v] = dist[u] + graph[u][v];
			});
		});

	}

	// print the constructed distance array
	//printSolution(dist);
}

// driver program to test above function
int main()
{

	/* Let us create the example graph discussed above */
	//int graph[V][V] = { { 0, 4, 0, 0, 0, 0, 0, 8, 0 },
	//					{ 4, 0, 8, 0, 0, 0, 0, 11, 0 },
	//					{ 0, 8, 0, 7, 0, 4, 0, 0, 2 },
	//					{ 0, 0, 7, 0, 9, 14, 0, 0, 0 },
	//					{ 0, 0, 0, 9, 0, 10, 0, 0, 0 },
	//					{ 0, 0, 4, 14, 10, 0, 2, 0, 0 },
	//					{ 0, 0, 0, 0, 0, 2, 0, 1, 6 },
	//					{ 8, 11, 0, 0, 0, 0, 1, 0, 7 },
	//					{ 0, 0, 2, 0, 0, 0, 6, 7, 0 } };

	queue q{ cpu_selector{} };
	buffer<int, 2> graph(range(V, V));

	dijkstra(graph, 0, q);

	return 0;
}