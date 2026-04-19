#include "skinning.h"
#include "vec3d.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li


/*
 * Constructor for the Skinning class.
 * Loads the skinning weights from a file and stores them in the meshSkinningJoints and meshSkinningWeights arrays.
 * Also computes the number of joints influencing each vertex.
 */
Skinning::Skinning(int numMeshVertices, const double * restMeshVertexPositions,
    const std::string & meshSkinningWeightsFilename)
{
  this->numMeshVertices = numMeshVertices; // store the number of vertices in the mesh
  this->restMeshVertexPositions = restMeshVertexPositions; // store the rest vertex positions

  cout << "Loading skinning weights..." << endl;
  ifstream fin(meshSkinningWeightsFilename.c_str()); // open the file containing the skinning weights
  assert(fin);
  int numWeightMatrixRows = 0, numWeightMatrixCols = 0;
  fin >> numWeightMatrixRows >> numWeightMatrixCols; // read the number of rows and columns from the file
  assert(fin.fail() == false);
  assert(numWeightMatrixRows == numMeshVertices); // check that the number of rows is equal to the number of vertices in the mesh
  int numJoints = numWeightMatrixCols; // store the number of joints

  vector<vector<int>> weightMatrixColumnIndices(numWeightMatrixRows);
  vector<vector<double>> weightMatrixEntries(numWeightMatrixRows);
  fin >> ws;
  while(fin.eof() == false)
  {
    int rowID = 0, colID = 0;
    double w = 0.0;
    fin >> rowID >> colID >> w;
    weightMatrixColumnIndices[rowID].push_back(colID);
    weightMatrixEntries[rowID].push_back(w);
    assert(fin.fail() == false);
    fin >> ws;
  }
  fin.close();

  // Build skinning joints and weights.
  numJointsInfluencingEachVertex = 0;
  // find the maximum number of joints influencing each vertex
  for (int i = 0; i < numMeshVertices; i++)
    numJointsInfluencingEachVertex = std::max(numJointsInfluencingEachVertex, (int)weightMatrixEntries[i].size());
  assert(numJointsInfluencingEachVertex >= 2);

  // Copy skinning weights from SparseMatrix into meshSkinningJoints and meshSkinningWeights.
  meshSkinningJoints.assign(numJointsInfluencingEachVertex * numMeshVertices, 0);
  meshSkinningWeights.assign(numJointsInfluencingEachVertex * numMeshVertices, 0.0);
  for (int vtxID = 0; vtxID < numMeshVertices; vtxID++)
  {
    vector<pair<double, int>> sortBuffer(numJointsInfluencingEachVertex);
    for (size_t j = 0; j < weightMatrixEntries[vtxID].size(); j++)
    {
      int frameID = weightMatrixColumnIndices[vtxID][j];
      double weight = weightMatrixEntries[vtxID][j];
      sortBuffer[j] = make_pair(weight, frameID);
    }
    sortBuffer.resize(weightMatrixEntries[vtxID].size());
    assert(sortBuffer.size() > 0);
    sort(sortBuffer.rbegin(), sortBuffer.rend()); // sort in descending order using reverse_iterators
    for(size_t i = 0; i < sortBuffer.size(); i++)
    {
      meshSkinningJoints[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].second;
      meshSkinningWeights[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].first;
    }

    // Note: When the number of joints used on this vertex is smaller than numJointsInfluencingEachVertex,
    // the remaining empty entries are initialized to zero due to vector::assign(XX, 0.0) .
  }
}

void Skinning::applySkinning(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const
{
  // Linear Blend Skinning:
  //   p' = sum_i ( w_i * M_i * p_rest )
  // where M_i = jointSkinTransforms[jointIndex_i] = globalTransform * globalRestTransform^{-1}
  // and the weights w_i sum to 1 for each vertex.
  for (int v = 0; v < numMeshVertices; v++)
  {
    Vec3d pRest(restMeshVertexPositions[3 * v + 0],
                restMeshVertexPositions[3 * v + 1],
                restMeshVertexPositions[3 * v + 2]);

    Vec3d pNew(0.0, 0.0, 0.0);
    for (int k = 0; k < numJointsInfluencingEachVertex; k++)
    {
      int idx = v * numJointsInfluencingEachVertex + k;
      double w = meshSkinningWeights[idx];
      if (w == 0.0) continue; // skip unused/zero-weight slots
      int jointID = meshSkinningJoints[idx];
      pNew += w * jointSkinTransforms[jointID].transformPoint(pRest);
    }

    newMeshVertexPositions[3 * v + 0] = pNew[0];
    newMeshVertexPositions[3 * v + 1] = pNew[1];
    newMeshVertexPositions[3 * v + 2] = pNew[2];
  }
}

