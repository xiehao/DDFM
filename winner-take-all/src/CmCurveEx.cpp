#include "curve-utils.hh"
#include "CmCurveEx.h"

using namespace std;
using namespace cv;

float const static PI_QUARTER = PI_FLOAT * 0.25f; 
float const static PI_EIGHTH = PI_FLOAT * 0.125f;

CmCurveEx::CmCurveEx(const Mat& srcImg1f, float maxOrntDif)
    : m_maxAngDif(maxOrntDif)
    , m_img1f(srcImg1f)
{
    CV_Assert(srcImg1f.type() == CV_32FC1);
    m_h = m_img1f.rows;
    m_w = m_img1f.cols;

    m_pDer1f.create(m_h, m_w, CV_32FC1);
    m_pOrnt1f.create(m_h, m_w, CV_32FC1);
    m_pLabel1i.create(m_h, m_w, CV_32SC1);
    m_pNext1i.create(m_h, m_w, CV_32SC1);
}

/* Compute the eigenvalues and eigenvectors of the Hessian matrix given by
dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
their absolute values. */
void CmCurveEx::compute_eigenvals(double dfdrr, double dfdrc, double dfdcc,
                                  double eigval[2], double eigvec[2][2])
{
    double theta, t, c, s, e1, e2, n1, n2; /* , phi; */

    /* Compute the eigenvalues and eigenvectors of the Hessian matrix. */
    if (dfdrc != 0.0) {
        theta = 0.5*(dfdcc-dfdrr)/dfdrc;
        t = 1.0/(fabs(theta)+sqrt(theta*theta+1.0));
        if (theta < 0.0) t = -t;
        c = 1.0/sqrt(t*t+1.0);
        s = t*c;
        e1 = dfdrr-t*dfdrc;
        e2 = dfdcc+t*dfdrc;
    } else {
        c = 1.0;
        s = 0.0;
        e1 = dfdrr;
        e2 = dfdcc;
    }
    n1 = c;
    n2 = -s;

    /* If the absolute value of an eigenvalue is larger than the other, put that
 eigenvalue into first position.  If both are of equal absolute value, put
 the negative one first. */
    if (fabs(e1) > fabs(e2)) {
        eigval[0] = e1;
        eigval[1] = e2;
        eigvec[0][0] = n1;
        eigvec[0][1] = n2;
        eigvec[1][0] = -n2;
        eigvec[1][1] = n1;
    } else if (fabs(e1) < fabs(e2)) {
        eigval[0] = e2;
        eigval[1] = e1;
        eigvec[0][0] = -n2;
        eigvec[0][1] = n1;
        eigvec[1][0] = n1;
        eigvec[1][1] = n2;
    } else {
        if (e1 < e2) {
            eigval[0] = e1;
            eigval[1] = e2;
            eigvec[0][0] = n1;
            eigvec[0][1] = n2;
            eigvec[1][0] = -n2;
            eigvec[1][1] = n1;
        } else {
            eigval[0] = e2;
            eigval[1] = e1;
            eigvec[0][0] = -n2;
            eigvec[0][1] = n1;
            eigvec[1][0] = n1;
            eigvec[1][1] = n2;
        }
    }
}

const Mat& CmCurveEx::CalSecDer(int kSize, float linkEndBound, float linkStartBound)
{
    Mat dxx, dxy, dyy;
    Sobel(m_img1f, dxx, CV_32F, 2, 0, kSize);
    Sobel(m_img1f, dxy, CV_32F, 1, 1, kSize);
    Sobel(m_img1f, dyy, CV_32F, 0, 2, kSize);

    double eigval[2], eigvec[2][2];
    for (int y = 0; y < m_h; y++)
    {
        float *xx = dxx.ptr<float>(y);
        float *xy = dxy.ptr<float>(y);
        float *yy = dyy.ptr<float>(y);
        float *pOrnt = m_pOrnt1f.ptr<float>(y);
        float *pDer = m_pDer1f.ptr<float>(y);
        for (int x = 0; x < m_w; x++)
        {
            compute_eigenvals(yy[x], xy[x], xx[x], eigval, eigvec);
            pOrnt[x] = (float)atan2(-eigvec[0][1], eigvec[0][0]); //㷨߷
            if (pOrnt[x] < 0.0f)
                pOrnt[x] += PI2;
            pDer[x] = float(eigval[0] > 0.0f ? eigval[0] : 0.0f);//׵
        }
    }

    GaussianBlur(m_pDer1f, m_pDer1f, Size(3, 3), 0);
    normalize(m_pDer1f, m_pDer1f, 0, 1, NORM_MINMAX);
    NoneMaximalSuppress(linkEndBound, linkStartBound);
    return m_pDer1f;
}

const Mat& CmCurveEx::CalFirDer(int kSize, float linkEndBound, float linkStartBound)
{
    Mat dxMat, dyMat;
    Sobel(m_img1f, dxMat, CV_32F, 1, 0, kSize);
    Sobel(m_img1f, dyMat, CV_32F, 0, 1, kSize);

    for (int y = 0; y < m_h; y++)
    {
        float *dx = dxMat.ptr<float>(y);
        float *dy = dyMat.ptr<float>(y);
        float *pOrnt = m_pOrnt1f.ptr<float>(y);
        float *pDer = m_pDer1f.ptr<float>(y);
        for (int x = 0; x < m_w; x++)
        {
            pOrnt[x] = (float)atan2f(dx[x], -dy[x]);
            if (pOrnt[x] < 0.0f)
                pOrnt[x] += PI2;

            pDer[x] = sqrt(dx[x]*dx[x] + dy[x]*dy[x]);
        }
    }

    GaussianBlur(m_pDer1f, m_pDer1f, Size(3, 3), 0);
    normalize(m_pDer1f, m_pDer1f, 0, 1, NORM_MINMAX);
    NoneMaximalSuppress(linkEndBound, linkStartBound);
    return m_pDer1f;
}

void CmCurveEx::NoneMaximalSuppress(float linkEndBound, float linkStartBound)
{
    CV_Assert(m_pDer1f.data != NULL && m_pLabel1i.data != NULL);

    m_StartPnt.clear();
    m_StartPnt.reserve(int(0.08 * m_h * m_w));
    PntImp linePoint;

    m_pLabel1i = IND_BG;
    for (int r = 1; r < m_h-1; r++)
    {
        float* pDer = m_pDer1f.ptr<float>(r);
        float* pOrnt = m_pOrnt1f.ptr<float>(r);
        int* pLineInd = m_pLabel1i.ptr<int>(r);
        for (int c = 1; c < m_w-1; c++)
        {
            if (pDer[c] < linkEndBound)
                continue;

            float cosN = sin(pOrnt[c]);
            float sinN = -cos(pOrnt[c]);
            int xSgn = CmSgn<float>(cosN);
            int ySgn = CmSgn<float>(sinN);
            cosN *= cosN;
            sinN *= sinN;

            if (pDer[c] >= (pDer[c + xSgn] * cosN +
                            m_pDer1f.at<float>(r + ySgn, c) * sinN)
                && pDer[c] >= (pDer[c - xSgn] * cosN +
                               m_pDer1f.at<float>(r - ySgn, c) * sinN))
            {
                pLineInd[c] = IND_NMS;
                if (pDer[c] < linkStartBound)
                    continue;

                //add to m_vStartPoint
                linePoint.second = Point(c, r);
                linePoint.first = pDer[c];
                m_StartPnt.push_back(linePoint);
            }
        }
    }
}

const CmEdges& CmCurveEx::Link(int shortRemoveBound /* = 3 */)
{
    CV_Assert(m_pDer1f.data != NULL && m_pLabel1i.data != NULL);

    sort(m_StartPnt.begin(), m_StartPnt.end(), linePointGreater);

    m_pNext1i = -1;
    m_vEdge.clear();
    m_vEdge.reserve(int(0.01 * m_w * m_h));
    CEdge crtEdge(0);//ǰ
    for (vector<PntImp>::iterator it = m_StartPnt.begin(); it != m_StartPnt.end(); it++)
    {
        Point pnt = it->second;
        if (m_pLabel1i.at<int>(pnt) != IND_NMS)
            continue;

        findEdge(pnt, crtEdge, false);
        findEdge(pnt, crtEdge, true);
        if (crtEdge.pointNum <= shortRemoveBound) {
            Point point = crtEdge.start;
            int i, nextInd;
            for (i = 1; i < crtEdge.pointNum; i++) {
                m_pLabel1i.at<int>(point) = IND_SR;
                nextInd = m_pNext1i.at<int>(point);
                point += DIRECTION8[nextInd];
            }
            m_pLabel1i.at<int>(point) = IND_SR;
        }
        else
        {
            m_vEdge.push_back(crtEdge);
            crtEdge.index++;
        }
    }

    // Get edge information
    int edgNum = (int)m_vEdge.size();
    for (int i = 0; i < edgNum; i++)
    {
        CEdge &edge = m_vEdge[i];
        vector<Point> &pnts = edge.pnts;
        pnts.resize(edge.pointNum);
        pnts[0] = edge.start;
        for (int j = 1; j < edge.pointNum; j++)
            pnts[j] = pnts[j-1] + DIRECTION8[m_pNext1i.at<int>(pnts[j-1])];
    }
    return m_vEdge;
}

/************************************************************************/
/* isForwardΪTRUEm_pOrntѰcrtEdge, ;m_pNext */
/* ӦֵΪ;ķֵ, ͬʱm_pLineIndֵΪǰߵı,Ҳ  */
/* һʱһΪcrtEdgeEnd.                  */
/* isForwardΪFALSEm_pOrntѰcrtEdge, ;     */
/* m_pNextӦֵΪ;ķֵ, ͬʱm_pLineIndֵΪǰߵ  */
/* .ҲһʱpointNum̫СactiveΪfalseƳ.  */
/* һΪcrtEdgeEnd.                              */
/************************************************************************/
void CmCurveEx::findEdge(Point seed, CEdge &crtEdge, bool isBackWard)
{
    Point pnt = seed;

    float ornt = m_pOrnt1f.at<float>(pnt);
    if (isBackWard){
        ornt += PI_FLOAT;
        if (ornt >= PI2)
            ornt -= PI2;
    }
    else{
        crtEdge.pointNum = 1;
        m_pLabel1i.at<int>(pnt) = crtEdge.index;
    }

    int orntInd, nextInd1, nextInd2;
    while (true) {
        /*************ȼѰһ㣬ϴ󲻼**************/
        //һDIRECTION16ѷ
        orntInd = int(ornt/PI_EIGHTH + 0.5f) % 16;
        if (jumpNext(pnt, ornt, crtEdge, orntInd, isBackWard))
            continue;
        //һDIRECTION8ѷ
        orntInd = int(ornt/PI_QUARTER + 0.5f) % 8;
        if (goNext(pnt, ornt, crtEdge, orntInd, isBackWard))
            continue;
        //һDIRECTION16ŷ
        orntInd = int(ornt/PI_EIGHTH + 0.5f) % 16;
        nextInd1 = (orntInd + 1) % 16;
        nextInd2 = (orntInd + 15) % 16;
        if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
            if(jumpNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
            if(jumpNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
        }
        else{//һDIRECTION16һ
            if(jumpNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
            if(jumpNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
        }
        //һDIRECTION8ŷ
        orntInd = int(ornt/PI_QUARTER + 0.5f) % 8;
        nextInd1 = (orntInd + 1) % 8;
        nextInd2 = (orntInd + 7) % 8;
        if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
            if(goNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
            if(goNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
        }
        else{//һDIRECTION8һ
            if(goNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
            if(goNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
        }


        /*************ȼѰһ㣬ϴҲ**************/
        //һDIRECTION16ѷ
        orntInd = int(ornt/PI_EIGHTH + 0.5f) % 16;
        if (jumpNext(pnt, ornt, crtEdge, orntInd, isBackWard))
            continue;
        //һDIRECTION8ѷ
        orntInd = int(ornt/PI_QUARTER + 0.5f) % 8;
        if (goNext(pnt, ornt, crtEdge, orntInd, isBackWard))
            continue;
        //һDIRECTION16ŷ
        orntInd = int(ornt/PI_EIGHTH + 0.5f) % 16;
        nextInd1 = (orntInd + 1) % 16;
        nextInd2 = (orntInd + 15) % 16;
        if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
            if(jumpNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
            if(jumpNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
        }
        else{//һDIRECTION16һ
            if(jumpNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
            if(jumpNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
        }
        //һDIRECTION8ŷ
        orntInd = int(ornt/PI_QUARTER + 0.5f) % 8;
        nextInd1 = (orntInd + 1) % 8;
        nextInd2 = (orntInd + 7) % 8;
        if (angle(DRT_ANGLE[nextInd1], ornt) < angle(DRT_ANGLE[nextInd2], ornt)) {
            if(goNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
            if(goNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
        }
        else{//һDIRECTION8һ
            if(goNext(pnt, ornt, crtEdge, nextInd2, isBackWard))
                continue;
            if(goNext(pnt, ornt, crtEdge, nextInd1, isBackWard))
                continue;
        }

        break;//ornt϶ûеĻ,Ѱ
    }

    if (isBackWard)
        crtEdge.start = pnt;
    else
        crtEdge.end = pnt;
}


float CmCurveEx::angle(float ornt1, float orn2)
{//ornt[0, 2*PI)֮, ֵ[0, PI/2)֮
    float agl = ornt1 - orn2;
    if (agl < 0)
        agl += PI2;
    if (agl >= PI_FLOAT)
        agl -= PI_FLOAT;
    if (agl >= PI_HALF)
        agl -= PI_FLOAT;
    return fabs(agl);
}

void CmCurveEx::refreshOrnt(float& ornt, float& newOrnt)
{
    static const float weightOld = 0.0f;
    static const float weightNew = 1.0f - weightOld;

    static const float largeBound = PI_FLOAT + PI_HALF;
    static const float smallBound = PI_HALF;

    if (newOrnt >= ornt + largeBound){
        newOrnt -= PI2;
        ornt = ornt * weightOld + newOrnt * weightNew;
        if (ornt < 0.0f)
            ornt += PI2;
    }
    else if (newOrnt + largeBound <= ornt){
        newOrnt += PI2;
        ornt = ornt * weightOld + newOrnt * weightNew;
        if (ornt >= PI2)
            ornt -= PI2;
    }
    else if (newOrnt >= ornt + smallBound){
        newOrnt -= PI_FLOAT;
        ornt = ornt * weightOld + newOrnt * weightNew;
        if (ornt < 0.0f)
            ornt += PI2;
    }
    else if(newOrnt + smallBound <= ornt){
        newOrnt += PI_FLOAT;
        ornt = ornt * weightOld + newOrnt * weightNew;
        if (ornt >= PI2)
            ornt -= PI2;
    }
    else
        ornt = ornt * weightOld + newOrnt * weightNew;
    newOrnt = ornt;
}

bool CmCurveEx::goNext(Point &pnt, float &ornt, CEdge &crtEdge, int orntInd, bool isBackward)
{
    Point pntN = pnt + DIRECTION8[orntInd];
    int &label = m_pLabel1i.at<int>(pntN);

    //õ㷽뵱ǰ߷Ƚϴ򲻼/***********һɱֵ**********************/
    if (CHK_IND(pntN) && (label == IND_NMS || label == IND_SR)) {
        if (angle(ornt, m_pOrnt1f.at<float>(pntN)) > m_maxAngDif)
            return false;

        label = crtEdge.index;
        if (isBackward)
            m_pNext1i.at<int>(pntN) = (orntInd + 4) % 8;
        else
            m_pNext1i.at<int>(pnt) = orntInd;
        crtEdge.pointNum++;

        //߷
        refreshOrnt(ornt, m_pOrnt1f.at<float>(pntN));
        pnt = pntN;
        return true;
    }
    return false;
}

bool CmCurveEx::jumpNext(Point &pnt, float &ornt, CEdge &crtEdge, int orntInd, bool isBackward)
{
    Point pnt2 = pnt + DIRECTION16[orntInd];
    if (CHK_IND(pnt2) && m_pLabel1i.at<int>(pnt2) <= IND_NMS) {
        if (angle(ornt, m_pOrnt1f.at<float>(pnt2)) > m_maxAngDif) //õ㷽뵱ǰ߷Ƚϴ򲻼
            return false;

        // DIRECTION16ϵorntInd൱DIRECTION8orntInd1,orntInd2
        // ĵ,orntInd = orntInd1 + orntInd2.˴ѡʹϵĵ
        // IND_NMSǵķϡ(orntInd1,orntInd2floor(orntInd/2)
        // ceil(orntInd/2)ѡ
        int orntInd1 = orntInd >> 1, orntInd2;
        Point pnt1 = pnt + DIRECTION8[orntInd1];
        if (m_pLabel1i.at<int>(pnt1) >= IND_BG && orntInd % 2) {
            orntInd1 = ((orntInd + 1) >> 1) % 8;
            pnt1 = pnt + DIRECTION8[orntInd1];
        }

        int &lineIdx1 = m_pLabel1i.at<int>(pnt1);
        if (lineIdx1 != -1) //ǰnPos1Ϊϵĵ㣬ܹ뵱ǰ
            return false;

        orntInd2 = orntInd - orntInd1;
        orntInd2 %= 8;

        lineIdx1 = crtEdge.index;
        m_pLabel1i.at<int>(pnt2) = crtEdge.index;
        if (isBackward) {
            m_pNext1i.at<int>(pnt1) = (orntInd1 + 4) % 8;
            m_pNext1i.at<int>(pnt2) = (orntInd2 + 4) % 8;
        }
        else{
            m_pNext1i.at<int>(pnt) = orntInd1;
            m_pNext1i.at<int>(pnt1) = orntInd2;
        }
        crtEdge.pointNum += 2;

        refreshOrnt(ornt, m_pOrnt1f.at<float>(pnt1));
        refreshOrnt(ornt, m_pOrnt1f.at<float>(pnt2));
        pnt = pnt2;
        return true;
    }
    return false;
}

void CmCurveEx::Demo(const Mat &img1u, bool isCartoon)
{
    Mat srcImg1f, show3u = Mat::zeros(img1u.size(), CV_8UC3);
    img1u.convertTo(srcImg1f, CV_32FC1, 1.0/255);

    CmCurveEx dEdge(srcImg1f);
    if (isCartoon)
        dEdge.CalSecDer();
    else
        dEdge.CalFirDer(5, 0.05f, 0.2f);

    dEdge.Link();
    const vector<CEdge> &edges = dEdge.GetEdges();

    for (size_t i = 0; i < edges.size(); i++)
    {
        Vec3b color(rand() % 255, rand() % 255, rand() % 255);
        const vector<Point> &pnts = edges[i].pnts;
        for (size_t j = 0; j < pnts.size(); j++)
            show3u.at<Vec3b>(pnts[j]) = color;
    }
    imshow("Derivatives", dEdge.GetDer());
    imshow("Image", img1u);
    imshow("Curve", show3u);
    waitKey(0);
    imwrite("ImageCurve.jpg", show3u);
}
