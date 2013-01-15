/* QPBO.cpp */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "instances.inc"


template <typename REAL> 
	void QPBO<REAL>::InitFreeList()
{
	Arc* a;
	Arc* a_last_free;

	first_free = a_last_free = NULL;
	for (a=arcs[0]; a<arc_max[0]; a+=2)
	if (!a->sister)
	{
		if (a_last_free) a_last_free->next = a;
		else        first_free = a;
		a_last_free = a;
	}
	if (a_last_free) a_last_free->next = NULL;
}

template <typename REAL> 
	QPBO<REAL>::QPBO(QPBO<REAL>& q)
	: node_num(q.node_num),
	  nodeptr_block(NULL),
	  changed_list(NULL),
	  fix_node_info_list(NULL),
	  stage(q.stage),
	  all_edges_submodular(q.all_edges_submodular),
	  error_function(q.error_function),
	  zero_energy(q.zero_energy)
{
	int node_num_max = q.node_shift/sizeof(Node);
	int arc_num_max = (int)(q.arc_max[0] - q.arcs[0]);
	Node* i;
	Arc* a;

	nodes[0] = (Node*) malloc(2*node_num_max*sizeof(Node));
	arcs[0] = (Arc*) malloc(2*arc_num_max*sizeof(Arc));
	if (!nodes[0] || !arcs[0]) { if (error_function) (*error_function)("Not enough memory!"); exit(1); }

	node_last[0] = nodes[0] + node_num;
	node_max[0] = nodes[1] = nodes[0] + node_num_max;
	node_last[1] = nodes[1] + node_num;
	node_max[1] = nodes[1] + node_num_max;
	node_shift = node_num_max*sizeof(Node);

	arc_max[0] = arcs[1] = arcs[0] + arc_num_max;
	arc_max[1] = arcs[1] + arc_num_max;
	arc_shift = arc_num_max*sizeof(Arc);

	maxflow_iteration = 0;

	memcpy(nodes[0], q.nodes[0], 2*node_num_max*sizeof(Node));
	memcpy(arcs[0], q.arcs[0], 2*arc_num_max*sizeof(Arc));

	for (i=nodes[0]; i<node_last[stage]; i++)
	{
		if (i==node_last[0]) i = nodes[1];
		if (i->first) i->first = (Arc*) ((char*)i->first + (((char*) arcs[0]) - ((char*) q.arcs[0])));
	}

	for (a=arcs[0]; a<arc_max[stage]; a++)
	{
		if (a == arc_max[0]) a = arcs[1];
		if (a->sister)
		{
			a->head              = (Node*) ((char*)a->head   + (((char*) nodes[0]) - ((char*) q.nodes[0])));
			if (a->next) a->next = (Arc*)  ((char*)a->next   + (((char*) arcs[0])  - ((char*) q.arcs[0])));
			a->sister            = (Arc*)  ((char*)a->sister + (((char*) arcs[0])  - ((char*) q.arcs[0])));
		}
	}

	InitFreeList();
}


template <typename REAL> 
	void QPBO<REAL>::reallocate_nodes(int node_num_max_new)
{
	code_assert(node_num_max_new > node_shift/((int)sizeof(Node)));
	Node* nodes_old[2] = { nodes[0], nodes[1] };

	int node_num_max = node_num_max_new;
	nodes[0] = (Node*) realloc(nodes_old[0], 2*node_num_max*sizeof(Node));
	if (!nodes[0]) { if (error_function) (*error_function)("Not enough memory!"); exit(1); }

	node_shift = node_num_max*sizeof(Node);
	node_last[0] = nodes[0] + node_num;
	node_max[0] = nodes[1] = nodes[0] + node_num_max;
	node_last[1] = nodes[1] + node_num;
	node_max[1] = nodes[1] + node_num_max;
	if (stage)
	{
		memmove(nodes[1], (char*)nodes[0] + ((char*)nodes_old[1] - (char*)nodes_old[0]), node_num*sizeof(Node));
	}

	Arc* a;
	for (a=arcs[0]; a<arc_max[stage]; a++)
	{
		if (a->sister)
		{
			int k = (a->head < nodes_old[1]) ? 0 : 1;
			a->head = (Node*) ((char*)a->head + (((char*) nodes[k]) - ((char*) nodes_old[k])));
		}
	}
}

template <typename REAL> 
	void QPBO<REAL>::reallocate_arcs(int arc_num_max_new)
{
	int arc_num_max_old = (int)(arc_max[0] - arcs[0]);
	int arc_num_max = arc_num_max_new; if (arc_num_max & 1) arc_num_max ++;
	code_assert(arc_num_max > arc_num_max_old);
	Arc* arcs_old[2] = { arcs[0], arcs[1] };

	arcs[0] = (Arc*) realloc(arcs_old[0], 2*arc_num_max*sizeof(Arc));
	if (!arcs[0]) { if (error_function) (*error_function)("Not enough memory!"); exit(1); }

	arc_shift = arc_num_max*sizeof(Arc);
	arc_max[0] = arcs[1] = arcs[0] + arc_num_max;
	arc_max[1] = arcs[1] + arc_num_max;

	if (stage)
	{
		memmove(arcs[1], arcs[0]+arc_num_max_old, arc_num_max_old*sizeof(Arc));
		memset(arcs[0]+arc_num_max_old, 0, (arc_num_max-arc_num_max_old)*sizeof(Arc));
		memset(arcs[1]+arc_num_max_old, 0, (arc_num_max-arc_num_max_old)*sizeof(Arc));
	}
	else
	{
		memset(arcs[0]+arc_num_max_old, 0, (2*arc_num_max-arc_num_max_old)*sizeof(Arc));
	}

	Node* i;
	Arc* a;
	for (i=nodes[0]; i<node_last[stage]; i++)
	{
		if (i==node_last[0]) i = nodes[1];

		if (i->first) 
		{
			int k = (i->first < arcs_old[1]) ? 0 : 1;
			i->first = (Arc*) ((char*)i->first + (((char*) arcs[k]) - ((char*) arcs_old[k])));
		}
	}
	for (a=arcs[0]; a<arc_max[stage]; a++)
	{
		if (a->sister)
		{
			if (a->next) 
			{
				int k = (a->next < arcs_old[1]) ? 0 : 1;
				a->next = (Arc*) ((char*)a->next + (((char*) arcs[k]) - ((char*) arcs_old[k])));
			}
			int k = (a->sister < arcs_old[1]) ? 0 : 1;
			a->sister = (Arc*) ((char*)a->sister + (((char*) arcs[k]) - ((char*) arcs_old[k])));
		}
	}

	InitFreeList();
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

template <typename REAL> 
	bool QPBO<REAL>::Save(char* filename, int format)
{
	int e;
	int edge_num = 0;
	for (e=GetNextEdgeId(-1); e>=0; e=GetNextEdgeId(e)) edge_num ++;

	if (format == 0)
	{
		FILE* fp;
		REAL E0, E1, E00, E01, E10, E11;
		int i, j;
		char* type_name;
		char* type_format;
		char FORMAT_LINE[64];
		int factor = (stage == 0) ? 2 : 1;

		get_type_information(type_name, type_format);

		fp = fopen(filename, "w");
		if (!fp) return false;

		fprintf(fp, "nodes=%d\n", GetNodeNum());
		fprintf(fp, "edges=%d\n", edge_num);
		fprintf(fp, "labels=2\n");
		fprintf(fp, "type=%s\n", type_name);
		fprintf(fp, "\n");

		sprintf(FORMAT_LINE, "n %%d\t%%%s %%%s\n", type_format, type_format);
		for (i=0; i<GetNodeNum(); i++)
		{
			GetTwiceUnaryTerm(i, E0, E1);
			REAL delta = (E0 < E1) ? E0 : E1;
			fprintf(fp, FORMAT_LINE, i, (E0-delta)/factor, (E1-delta)/factor);
		}
		sprintf(FORMAT_LINE, "e %%d %%d\t%%%s %%%s %%%s %%%s\n", type_format, type_format, type_format, type_format);
		for (e=GetNextEdgeId(-1); e>=0; e=GetNextEdgeId(e))
		{
			GetTwicePairwiseTerm(e, i, j, E00, E01, E10, E11);
			fprintf(fp, FORMAT_LINE, i, j, E00/factor, E01/factor, E10/factor, E11/factor);
		}
		fclose(fp);
		return true;
	}
	if (format == 1)
	{
		FILE* fp;
		REAL E0, E1, E00, E01, E10, E11;
		int i, j;
		Arc* a;
		char* type_name;
		char* type_format;

		if (stage == 0) Solve();

		get_type_information(type_name, type_format);
		if (type_format[0] != 'd') return false;

		fp = fopen(filename, "w");
		if (!fp) return false;

		fprintf(fp, "p %d %d\n", GetNodeNum(), GetNodeNum() + edge_num);

		for (i=0; i<GetNodeNum(); i++)
		{
			GetTwiceUnaryTerm(i, E0, E1);
			REAL delta = E1 - E0;
			for (a=nodes[0][i].first; a; a=a->next)
			{
				if (IsNode0(a->head)) delta += a->sister->r_cap + GetMate(a)->sister->r_cap;
				else                  delta -= a->r_cap + GetMate(a)->r_cap;
			}
			fprintf(fp, "1 %d %d\n", i+1, delta);
		}
		for (e=GetNextEdgeId(-1); e>=0; e=GetNextEdgeId(e))
		{
			GetTwicePairwiseTerm(e, i, j, E00, E01, E10, E11);
			fprintf(fp, "2 %d %d %d\n", i+1, j+1, E00 + E11 - E01 - E10);
		}
		fclose(fp);
		return true;
	}
	return false;
}

template <typename REAL> 
	bool QPBO<REAL>::Load(char* filename)
{
	FILE* fp;
	REAL E0, E1, E00, E01, E10, E11;
	int i, j;
	char* type_name;
	char* type_format;
	char LINE[256], FORMAT_LINE_NODE[64], FORMAT_LINE_EDGE[64];
	int NODE_NUM, EDGE_NUM, K;

	get_type_information(type_name, type_format);

	fp = fopen(filename, "r");
	if (!fp) { printf("Cannot open %s\n", filename); return false; }

	if (fscanf(fp, "nodes=%d\n", &NODE_NUM) != 1) { printf("%s: wrong format\n", filename); fclose(fp); return false; }
	if (fscanf(fp, "edges=%d\n", &EDGE_NUM) != 1) { printf("%s: wrong format\n", filename); fclose(fp); return false; }
	if (fscanf(fp, "labels=%d\n", &K) != 1) { printf("%s: wrong format\n", filename); fclose(fp); return false; }
	if (K != 2) { printf("%s: wrong number of labels\n", filename); fclose(fp); return false; }
	if (fscanf(fp, "type=%10s\n", LINE) != 1) { printf("%s: wrong format\n", filename); fclose(fp); return false; }
	if (strcmp(LINE, type_name)) { printf("%s: type REAL mismatch\n", filename); fclose(fp); return false; }

	Reset();

	AddNode(NODE_NUM+4);
	node_num -= 4;
	node_last[0] -= 4;
	node_last[1] -= 4;

	sprintf(FORMAT_LINE_NODE, "n %%d %%%s %%%s\n", type_format, type_format);
	sprintf(FORMAT_LINE_EDGE, "e %%d %%d %%%s %%%s %%%s %%%s\n", type_format, type_format, type_format, type_format);
	while (fgets(LINE, sizeof(LINE), fp))
	{
		if (sscanf(LINE, FORMAT_LINE_EDGE, &i, &j, &E00, &E01, &E10, &E11) == 6)
		{
			if (i<0 || i>=NODE_NUM || j<0 || j>=NODE_NUM || i==j) { printf("%s: wrong format\n", filename); fclose(fp); return false; }
			AddPairwiseTerm(i, j, E00, E01, E10, E11);
		}
		else if (sscanf(LINE, FORMAT_LINE_NODE, &i, &E0, &E1) == 3)
		{
			if (i<0 || i>=NODE_NUM) { printf("%s: wrong format\n", filename); fclose(fp); return false; }
			AddUnaryTerm(i, E0, E1);
		}
	}

	fclose(fp);
	return true;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


template <typename REAL> 
	void QPBO<REAL>::TransformToSecondStage(bool copy_trees)
{
	// add non-submodular edges
	Node* i[2];
	Node* j[2];
	Arc* a[2];

	memset(nodes[1], 0, node_num*sizeof(Node));
	node_last[1] = nodes[1] + node_num;

	if (!copy_trees)
	{
		for (i[0]=nodes[0], i[1]=nodes[1]; i[0]<node_last[0]; i[0]++, i[1]++)
		{
			i[1]->first = NULL;
			i[1]->tr_cap = -i[0]->tr_cap;
		}

		for (a[0]=arcs[0], a[1]=arcs[1]; a[0]<arc_max[0]; a[0]+=2, a[1]+=2)
		{
			if (!a[0]->sister) continue;

			code_assert(IsNode0(a[0]->sister->head));
			SET_SISTERS(a[1], a[1]+1);
			if (IsNode0(a[0]->head))
			{
				i[1] = GetMate0(a[0]->sister->head);
				j[1] = GetMate0(a[0]->head);

				SET_FROM(a[1],         j[1]);
				SET_FROM(a[1]->sister, i[1]);
				SET_TO(a[1],         i[1]);
				SET_TO(a[1]->sister, j[1]);
			}
			else
			{
				i[0] = a[0]->sister->head;
				i[1] = GetMate0(i[0]);
				j[1] = a[0]->head;
				j[0] = GetMate1(j[1]);

				SET_FROM(a[0],         i[0]);
				SET_FROM(a[0]->sister, j[1]);
				SET_FROM(a[1],         j[0]);
				SET_FROM(a[1]->sister, i[1]);
				SET_TO(a[1],         i[1]);
				SET_TO(a[1]->sister, j[0]);
			}
			a[1]->r_cap = a[0]->r_cap;
			a[1]->sister->r_cap = a[0]->sister->r_cap;
		}
	}
	else
	{
		for (i[0]=nodes[0], i[1]=nodes[1]; i[0]<node_last[0]; i[0]++, i[1]++)
		{
			i[1]->first = NULL;
			i[1]->tr_cap = -i[0]->tr_cap;
			i[1]->is_sink = i[0]->is_sink ^ 1;
			i[1]->DIST = i[0]->DIST;
			i[1]->TS = i[0]->TS;

			if (i[0]->parent == NULL || i[0]->parent == QPBO_MAXFLOW_TERMINAL) i[1]->parent = i[0]->parent;
			else i[1]->parent = GetMate0(i[0]->parent->sister);
		}

		for (a[0]=arcs[0], a[1]=arcs[1]; a[0]<arc_max[0]; a[0]+=2, a[1]+=2)
		{
			if (!a[0]->sister) continue;

			code_assert(IsNode0(a[0]->sister->head));
			SET_SISTERS(a[1], a[1]+1);
			if (IsNode0(a[0]->head))
			{
				i[1] = GetMate0(a[0]->sister->head);
				j[1] = GetMate0(a[0]->head);

				SET_FROM(a[1],         j[1]);
				SET_FROM(a[1]->sister, i[1]);
				SET_TO(a[1],         i[1]);
				SET_TO(a[1]->sister, j[1]);
			}
			else
			{
				i[0] = a[0]->sister->head;
				i[1] = GetMate0(i[0]);
				j[1] = a[0]->head;
				j[0] = GetMate1(j[1]);

				SET_FROM(a[0],         i[0]);
				SET_FROM(a[0]->sister, j[1]);
				SET_FROM(a[1],         j[0]);
				SET_FROM(a[1]->sister, i[1]);
				SET_TO(a[1],         i[1]);
				SET_TO(a[1]->sister, j[0]);

				mark_node(i[0]);
				mark_node(i[1]);
				mark_node(j[0]);
				mark_node(j[1]);
			}
			a[1]->r_cap = a[0]->r_cap;
			a[1]->sister->r_cap = a[0]->sister->r_cap;
		}
	}

	stage = 1;
}

template <typename REAL>
	void QPBO<REAL>::MergeParallelEdges()
{
	if (stage == 0) TransformToSecondStage(false);
	Node* i;
	Node* j;
	Arc* a;
	Arc* a_next;

	for (i=nodes[0]; i<node_last[0]; i++)
	{
		for (a=i->first; a; a=a->next)
		{
			j = a->head;
			if (!IsNode0(j)) j = GetMate1(j);
			j->parent = a;
		}
		for (a=i->first; a; a=a_next)
		{
			a_next = a->next;
			j = a->head;
			if (!IsNode0(j)) j = GetMate1(j);
			if (j->parent == a) continue;
			if (MergeParallelEdges(j->parent, a)==0) 
			{
				j->parent = a;
				a_next = a->next;
			}
		}
	}
}

template <typename REAL>
    REAL QPBO<REAL>::ComputeTwiceEnergy(int option)
{
	REAL E = 2*zero_energy, E1[2], E2[2][2];
	int i, j, e;
	int xi, xj;

	for (i=0; i<GetNodeNum(); i++)
	{
		GetTwiceUnaryTerm(i, E1[0], E1[1]);
		if (option == 0) xi = (nodes[0][i].label < 0) ? 0 : nodes[0][i].label;
		else             xi = nodes[0][i].user_label;
		code_assert(xi==0 || xi==1);
		E += E1[xi] - E1[0];
	}
	for (e=GetNextEdgeId(-1); e>=0; e=GetNextEdgeId(e))
	{
		GetTwicePairwiseTerm(e, i, j, E2[0][0], E2[0][1], E2[1][0], E2[1][1]);
		if (option == 0)
		{
			xi = (nodes[0][i].label < 0) ? 0 : nodes[0][i].label;
			xj = (nodes[0][j].label < 0) ? 0 : nodes[0][j].label;
		}
		else
		{
			xi = nodes[0][i].user_label;
			xj = nodes[0][j].user_label;
		}
		E += E2[xi][xj] - E2[0][0];
	}
	return E;
}

template <typename REAL>
    REAL QPBO<REAL>::ComputeTwiceEnergy(int* solution)
{
	REAL E = 2*zero_energy, E1[2], E2[2][2];
	int i, j, e;

	for (i=0; i<GetNodeNum(); i++)
	{
		GetTwiceUnaryTerm(i, E1[0], E1[1]);
		if (solution[i] == 1) E += E1[1];
	}
	for (e=GetNextEdgeId(-1); e>=0; e=GetNextEdgeId(e))
	{
		GetTwicePairwiseTerm(e, i, j, E2[0][0], E2[0][1], E2[1][0], E2[1][1]);
		E += E2[(solution[i] == 1) ? 1 : 0][(solution[j] == 1) ? 1 : 0] - E2[0][0];
	}
	return E;
}

template <typename REAL>
	void QPBO<REAL>::TestRelaxedSymmetry()
{
	Node* i;
	Arc* a;
	REAL c1, c2;

	if (stage == 0) return;

	for (i=nodes[0]; i<node_last[0]; i++)
	{
		if (i->is_removed) continue;
		c1 = i->tr_cap;
		for (a=i->first; a; a=a->next) c1 += a->sister->r_cap;
		c2 = -GetMate0(i)->tr_cap;
		for (a=GetMate0(i)->first; a; a=a->next) c2 += a->r_cap;
		if (c1 != c2)
		{
			code_assert(0);
			exit(1);
		}
	}
}

#include "QPBO.h"
