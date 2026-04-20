#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  int numJoints = fk.getNumJoints();
  std::vector<Mat3<real>> globalR(numJoints);
  std::vector<Vec3<real>> globalT(numJoints);

  for (int k = 0; k < numJoints; k++)
  {
    int i = fk.getJointUpdateOrder(k);

    // Local rotation = jointOrientation (always XYZ) * eulerAngles (per-joint order)
    real angles[3];
    angles[0] = eulerAngles[i*3+0];
    angles[1] = eulerAngles[i*3+1];
    angles[2] = eulerAngles[i*3+2];
    Mat3<real> Re = Euler2Rotation(angles, fk.getJointRotateOrder(i));

    const Vec3d & jo = fk.getJointOrient(i);
    real joAngles[3];
    joAngles[0] = (real)jo[0];
    joAngles[1] = (real)jo[1];
    joAngles[2] = (real)jo[2];
    Mat3<real> Ro = Euler2Rotation(joAngles, RotateOrder::XYZ);

    Mat3<real> Rlocal = Ro * Re;
    const Vec3d & tv = fk.getJointRestTranslation(i);
    Vec3<real> tlocal((real)tv[0], (real)tv[1], (real)tv[2]);

    int parent = fk.getJointParent(i);
    if (parent < 0)
    {
      globalR[i] = Rlocal;
      globalT[i] = tlocal;
    }
    else
    {
      // f_global = f_parent ∘ f_local
      // R_out = R_parent * R_local,  t_out = R_parent * t_local + t_parent
      multiplyAffineTransform4ds(globalR[parent], globalT[parent], Rlocal, tlocal, globalR[i], globalT[i]);
    }
  }

  // Write out the world-space positions of the IK handle joints
  for (int j = 0; j < numIKJoints; j++)
  {
    int id = IKJointIDs[j];
    handlePositions[j*3+0] = globalT[id][0];
    handlePositions[j*3+1] = globalT[id][1];
    handlePositions[j*3+2] = globalT[id][2];
  }
}

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
  int n = FKInputDim;  // numJoints * 3
  int m = FKOutputDim; // numIKJoints * 3

  trace_on(adolc_tagID);

  vector<adouble> eulerAngles(n);
  for (int i = 0; i < n; i++)
    eulerAngles[i] <<= 0.0;

  vector<adouble> handlePositions(m);
  forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, eulerAngles, handlePositions);

  vector<double> output(m);
  for (int i = 0; i < m; i++)
    handlePositions[i] >>= output[i];

  trace_off();
}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
  int numJoints = fk->getNumJoints();
  int n = FKInputDim;  // numJoints * 3
  int m = FKOutputDim; // numIKJoints * 3

  // Flatten current Euler angles into a plain double array
  vector<double> theta(n);
  for (int i = 0; i < numJoints; i++)
  {
    theta[i*3+0] = jointEulerAngles[i][0];
    theta[i*3+1] = jointEulerAngles[i][1];
    theta[i*3+2] = jointEulerAngles[i][2];
  }

  // Evaluate FK to get current IK handle positions
  vector<double> currentPos(m);
  ::function(adolc_tagID, m, n, theta.data(), currentPos.data());

  // Δp = target - current
  Eigen::VectorXd dp(m);
  for (int j = 0; j < numIKJoints; j++)
  {
    dp[j*3+0] = targetHandlePositions[j][0] - currentPos[j*3+0];
    dp[j*3+1] = targetHandlePositions[j][1] - currentPos[j*3+1];
    dp[j*3+2] = targetHandlePositions[j][2] - currentPos[j*3+2];
  }

  // Compute Jacobian J (m × n, row-major)
  vector<double> jacobianFlat(m * n);
  vector<double*> jacobianRows(m);
  for (int i = 0; i < m; i++)
    jacobianRows[i] = &jacobianFlat[i * n];
  ::jacobian(adolc_tagID, m, n, theta.data(), jacobianRows.data());

  Eigen::MatrixXd J(m, n);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      J(i, j) = jacobianFlat[i * n + j];

  // Tikhonov regularization: solve (J^T J + α I) Δθ = J^T Δp
  const double alpha = 0.01;
  Eigen::MatrixXd A = J.transpose() * J + alpha * Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd b = J.transpose() * dp;
  Eigen::VectorXd dtheta = A.ldlt().solve(b);

  // Write updated Euler angles back
  for (int i = 0; i < numJoints; i++)
  {
    jointEulerAngles[i][0] += dtheta[i*3+0];
    jointEulerAngles[i][1] += dtheta[i*3+1];
    jointEulerAngles[i][2] += dtheta[i*3+2];
  }
}

