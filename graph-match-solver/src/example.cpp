#include <stdio.h>
#include "GraphMatching.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

const std::string workspace = "../../data/";

GraphMatching* LoadMatchingFile(char* filename, bool load_neib = true)
{
	FILE* fp;
	int N[2], A, E, _A = 0, _E = 0;
	char LINE[256];
	double cost;
	GraphMatching* m = NULL;

	fp = fopen(filename, "r");
	if (!fp) { printf("Can't open %s\n", filename); exit(1); }

	while (fgets(LINE, sizeof(LINE)-1, fp))
	{
		if (LINE[0] == 'p')
		{
			if (m || sscanf(&LINE[1], "%d %d %d %d\n", &N[0], &N[1], &A, &E) != 4) { printf("%s: wrong format1!\n", filename); exit(1); }
			m = new GraphMatching(N[0], N[1], A, E);
		}
		else if (LINE[0] == 'a')
		{
			int a, i0, i1;
			if (!m || sscanf(&LINE[1], "%d %d %d %Lf\n", &a, &i0, &i1, &cost) != 4 
			    || a!=_A++ || _A>A || i0<0 || i0>=N[0] || i1<0 || i1>=N[1]) { printf("%s: wrong format2!\n", filename); exit(1); }
			m->AddAssignment(i0, i1, cost);
		}
		else if (LINE[0] == 'e')
		{
			int a, b;
			if (!m || sscanf(&LINE[1], "%d %d %Lf\n", &a, &b, &cost) != 3
			    || (_E++)>=E || a<0 || a>=A || b<0 || b>=A || a==b) { printf("%s: wrong format3!\n", filename); exit(1); }
			m->AddEdge(a, b, cost);
		}
		else if (LINE[0] == 'n')
		{
			if (load_neib)
			{
				int r, i, j;
				if (!m || sscanf(&LINE[1], "%d %d %d\n", &r, &i, &j) != 3 
					|| r<0 || r>1 || i<0 || i>=N[r] || j<0 || j>=N[r] || i==j) { printf("%s: wrong format4!\n", filename); exit(1); }
				m->AddNeighbors(r, i, j);
			}
		}
	}
	fclose(fp);

	if (!m || _A!=A || _E!=E) { printf("%s: wrong format5!\n", filename); exit(1); }

	/////////////////////////////////////////////////////////////////////////

	printf("problem from %s loaded (N0=%d, N1=%d, A=%d, E=%d\n", filename, N[0], N[1], A, E);

	return m;
}

GraphMatching *load_problem(const std::string &_file_name,
                            bool load_neighbor = true)
{
    GraphMatching *m = NULL;

    std::ifstream file(_file_name.c_str());

    if (!file)
    {
        std::cerr << "Loading file failed!!" << std::endl;
        return NULL;
    }

    std::string line;
    int N0(0), N1(0), _A(0), _E(0), A(0), E(0);
    float cost(0);
    for (; std::getline(file, line);)
    {
        std::istringstream stream(line);
        std::string word;
        stream >> word;
        if ("c" == word)
        {
            /*!< comment */
            std::cout << word << std::endl;
        }
        else if ("p" == word)
        {
            /*!< problem description */
            stream >> N0 >> N1 >> A >> E;
            if (m || !(N0 && N1 && A && E))
            {
                std::cerr << "wrong format 1!!" << std::endl;
                break;
            }
            else
            {
                m = new GraphMatching(N0, N1, A, E);
            }
        }
        else if ("a" == word)
        {
            /*!< assignment of a pair */
            int a(0), i0(0), i1(0);
            stream >> a >> i0 >> i1 >> cost;
            if (!m || a != _A++ || _A > A || i0 < 0 || i1 > N0 || i1 < 0 ||
                    i1 > N1)
            {
                std::cerr << "wrong format 2!!" << std::endl;
                break;
            }
            else
            {
                m->AddAssignment(i0, i1, cost);
            }
        }
        else if ("e" == word)
        {
            /*!< edge between two assignments */
            int a(0), b(0);
            stream >> a >> b >> cost;
            if (!m || E <= (_E++) || a < 0 || a >=A || b < 0 || b >= A ||
                    a == b)
            {
                std::cerr << "wrong format 3!!" << std::endl;
                break;
            }
            else
            {
                m->AddEdge(a, b, cost);
            }
        }
        else if ("n0" == word && load_neighbor)
        {
            /*!< OPTIONAL: neighborhood of each side */
            int i(0), j(0);
            stream >> i >> j;
            if (!m || i < 0 || i >= N0 || j < 0 || j >= N0 || i == j)
            {
                std::cerr << "wrong format 4!!" << std::endl;
                break;
            }
            else
            {
                m->AddNeighbors(0, i, j);
            }
        }
        else if ("n1" == word && load_neighbor)
        {
            /*!< OPTIONAL: neighborhood of each side */
            int i(0), j(0);
            stream >> i >> j;
            if (!m || i < 0 || i >= N1 || j < 0 || j >= N1 || i == j)
            {
                std::cerr << "wrong format 4!!" << std::endl;
                break;
            }
            else
            {
                m->AddNeighbors(1, i, j);
            }
        }
    }

    file.close();

    if (!m || _A != A || _E != E)
    {
        std::cerr << "wrong format 5!!" << std::endl;
        return NULL;
    }

    std::cout << "Problem from " << _file_name << " loaded (N0 = " << N0
              << ", N1 = " << N1 << ", A = " << A << ", E = " << E << std::endl;

    return m;
}

int main(void)
{
    char *str = const_cast<char *>((workspace + "DATA.TXT").c_str());
    GraphMatching* m = LoadMatchingFile(str);
//    GraphMatching *m = load_problem(workspace + "DATA.TXT");

	// you may need to experiment a bit with what subproblems to add. 

	//m->AddLinearSubproblem();
	//m->AddMaxflowSubproblem();
	m->AddLocalSubproblems(3);
	//m->AddTreeSubproblems();

	m->SolveDD(10000, 1e-5);

    int n_N0 = m->GetN0();
    int *solution = m->GetSolution();
    std::ofstream file((workspace + "results.txt").c_str());
    file.clear();

    for (int i = 0; i < n_N0; ++i)
    {
        file << solution[i] << std::endl;
    }

    file.close();

	delete m;

	return 0;
}
