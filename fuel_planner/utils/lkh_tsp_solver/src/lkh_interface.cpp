#include <lkh_tsp_solver/lkh_interface.h>
#include <algorithm>
#include <fstream>

namespace lkh_interface{

int solveTSPLKH(const char* input_file) {
  GainType Cost, OldOptimum;
  double Time, LastTime = GetTime();

  /* Read the specification of the problem */
  // ParameterFileName =
  // "/home/boboyu/workspaces/plan_ws/src/fast_planner/utils/lkh_tsp_solver/resource/test.par";
  ParameterFileName = const_cast<char*>(input_file);
  ReadParameters();
  MaxMatrixDimension = 20000;
  MergeWithTour = Recombination == IPT ? MergeWithTourIPT : MergeWithTourGPX2;
  ReadProblem();

  if (SubproblemSize > 0) {
    if (DelaunayPartitioning)
      SolveDelaunaySubproblems();
    else if (KarpPartitioning)
      SolveKarpSubproblems();
    else if (KCenterPartitioning)
      SolveKCenterSubproblems();
    else if (KMeansPartitioning)
      SolveKMeansSubproblems();
    else if (RohePartitioning)
      SolveRoheSubproblems();
    else if (MoorePartitioning || SierpinskiPartitioning)
      SolveSFCSubproblems();
    else
      SolveTourSegmentSubproblems();
    return EXIT_SUCCESS;
  }
  AllocateStructures();
  CreateCandidateSet();
  InitializeStatistics();

  if (Norm != 0)
    BestCost = PLUS_INFINITY;
  else {
    /* The ascent has solved the problem! */
    Optimum = BestCost = (GainType)LowerBound;
    UpdateStatistics(Optimum, GetTime() - LastTime);
    RecordBetterTour();
    RecordBestTour();
    WriteTour(OutputTourFileName, BestTour, BestCost);
    WriteTour(TourFileName, BestTour, BestCost);
    Runs = 0;
  }

  /* Find a specified number (Runs) of local optima */
  for (Run = 1; Run <= Runs; Run++) {
    LastTime = GetTime();
    Cost = FindTour(); /* using the Lin-Kernighan heuristic */
    if (MaxPopulationSize > 1) {
      /* Genetic algorithm */
      int i;
      for (i = 0; i < PopulationSize; i++) {
        GainType OldCost = Cost;
        Cost = MergeTourWithIndividual(i);
        if (TraceLevel >= 1 && Cost < OldCost) {
          printff("  Merged with %d: Cost = " GainFormat, i + 1, Cost);
          if (Optimum != MINUS_INFINITY && Optimum != 0)
            printff(", Gap = %0.4f%%", 100.0 * (Cost - Optimum) / Optimum);
          printff("\n");
        }
      }
      if (!HasFitness(Cost)) {
        if (PopulationSize < MaxPopulationSize) {
          AddToPopulation(Cost);
          if (TraceLevel >= 1) PrintPopulation();
        } else if (Cost < Fitness[PopulationSize - 1]) {
          i = ReplacementIndividual(Cost);
          ReplaceIndividualWithTour(i, Cost);
          if (TraceLevel >= 1) PrintPopulation();
        }
      }
    } else if (Run > 1)
      Cost = MergeTourWithBestTour();
    if (Cost < BestCost) {
      BestCost = Cost;
      RecordBetterTour();
      RecordBestTour();
      WriteTour(OutputTourFileName, BestTour, BestCost);
      WriteTour(TourFileName, BestTour, BestCost);
    }
    OldOptimum = Optimum;
    if (Cost < Optimum) {
      if (FirstNode->InputSuc) {
        Node* N = FirstNode;
        while ((N = N->InputSuc = N->Suc) != FirstNode)
          ;
      }
      Optimum = Cost;
      printff("*** New optimum = " GainFormat " ***\n\n", Optimum);
    }
    Time = fabs(GetTime() - LastTime);
    UpdateStatistics(Cost, Time);
    if (TraceLevel >= 1 && Cost != PLUS_INFINITY) {
      // printff("Run %d: Cost = " GainFormat, Run, Cost);
      // if (Optimum != MINUS_INFINITY && Optimum != 0)
      //   printff(", Gap = %0.4f%%", 100.0 * (Cost - Optimum) / Optimum);
      // printff(", Time = %0.2f sec. %s\n\n", Time,
      //         Cost < Optimum ? "<" : Cost == Optimum ? "=" : "");
    }
    if (StopAtOptimum && Cost == OldOptimum && MaxPopulationSize >= 1) {
      Runs = Run;
      break;
    }
    if (PopulationSize >= 2 && (PopulationSize == MaxPopulationSize || Run >= 2 * MaxPopulationSize) &&
        Run < Runs) {
      Node* N;
      int Parent1, Parent2;
      Parent1 = LinearSelection(PopulationSize, 1.25);
      do
        Parent2 = LinearSelection(PopulationSize, 1.25);
      while (Parent2 == Parent1);
      ApplyCrossover(Parent1, Parent2);
      N = FirstNode;
      do {
        if (ProblemType != HCP && ProblemType != HPP) {
          int d = C(N, N->Suc);
          AddCandidate(N, N->Suc, d, INT_MAX);
          AddCandidate(N->Suc, N, d, INT_MAX);
        }
        N = N->InitialSuc = N->Suc;
      } while (N != FirstNode);
    }
    SRandom(++Seed);
  }
  PrintStatistics();
  return EXIT_SUCCESS;
}

void readTourFromFile(std::vector<int>& indices, const std::string& file_dir){
    // Read optimal tour from the tour section of result file
    std::ifstream res_file(file_dir);
    std::string res;
    int dimension;
    while (getline(res_file, res)) {
      if (res.compare(0, 9, "DIMENSION") == 0) {
        dimension = std::stoi(res.substr(12,3));
        }
      // Go to tour section
      if (res.compare("TOUR_SECTION") == 0) break;
    }

    if (false) {
      // Read path for Symmetric TSP formulation
      getline(res_file, res);  // Skip current pose
      getline(res_file, res);
      int id = stoi(res);
      bool rev = (id == dimension);  // The next node is virutal depot?

      while (id != -1) {
        indices.push_back(id - 2);
        getline(res_file, res);
        id = std::stoi(res);
      }
      if (rev) std::reverse(indices.begin(), indices.end());
      indices.pop_back();  // Remove the depot

    } else {
      // Read path for ATSP formulation
      while (getline(res_file, res)) {
        // Read indices of frontiers in optimal tour
        int id = std::stoi(res);
        if (id == 1)  // Ignore the current state
          continue;
        if (id == -1) break;
        indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
      }
    }

    res_file.close();
}

void writeCostMatToFile(const Eigen::MatrixXd& cost_mat, const std::string& file_dir){
    uint dimension = cost_mat.rows();
    // Write params and cost matrix to problem file
    std::ofstream prob_file(file_dir);
    // Problem specification part, follow the format of TSPLIB

    std::string prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + std::to_string(dimension) +
        "\nEDGE_WEIGHT_TYPE : "
        "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";

    prob_file << prob_spec;

    // Problem data part
    const int scale = 100;
    // Use Asymmetric TSP
    for (int i = 0; i < dimension; ++i) {
      for (int j = 0; j < dimension; ++j) {
        int int_cost = cost_mat(i, j) * scale;
        prob_file << int_cost << " ";
      }
      prob_file << "\n";
    }
    

    prob_file << "EOF";
    prob_file.close();
}

}
