/************************************************************************/
/*  This software is developed by Ming-Ming Cheng.				        */
/*       Url: http://cg.cs.tsinghua.edu.cn/people/~cmm/                 */
/*  This software is free fro non-commercial use. In order to use this	*/
/*  software for academic use, you must cite the corresponding paper:	*/
/*      Ming-Ming Cheng, Curve Structure Extraction for Cartoon Images, */
/*      in The 5th Joint Conference on Harmonious Human Machine			*/
/*      Environment (HHME), 2009, pp. 13-20.							*/
/************************************************************************/

#ifndef CURVE_UTILS_HH
#define CURVE_UTILS_HH

#include <stdio.h>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

extern cv::Point const DIRECTION4[4];
extern cv::Point const DIRECTION8[8];
extern cv::Point const DIRECTION16[16];
extern float const DRT_ANGLE[8];
extern float const PI_FLOAT;
extern float const PI2;
extern float const PI_HALF;

const double EPS = 1e-8;		// Epsilon (zero value)
#define CHK_IND(p) ((p).x >= 0 && (p).x < m_w && (p).y >= 0 && (p).y < m_h)

template<typename T>
inline int CmSgn(T number)
{
    if (fabs(number) < EPS)
        return 0;
    return number > 0 ? 1 : -1;
}

#endif // CURVE_UTILS_HH
