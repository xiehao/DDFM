#ifndef CMCURVEEX_HH
#define CMCURVEEX_HH

/************************************************************************/
/*  This software is developed by Ming-Ming Cheng.			*/
/*       Url: http://cg.cs.tsinghua.edu.cn/people/~cmm/                 */
/*  This software is free fro non-commercial use. In order to use this	*/
/*  software for academic use, you must cite the corresponding paper:	*/
/*      Ming-Ming Cheng, Curve Structure Extraction for Cartoon Images, */
/*		Proceedings of NCMT 2009, 1-8				*/
/************************************************************************/


class CmCurveEx
{
public:
    typedef struct CEdge{
        CEdge(int _index){ index = _index; }
        ~CEdge(void){}

        // Domains assigned during link();
        int index;    // Start from 0
        int pointNum;
        cv::Point start, end;
        std::vector<cv::Point> pnts;
    }CEdge;

    CmCurveEx(const cv::Mat& srcImg1f, float maxOrntDif = 0.25f * CV_PI);

    // Input kernel size for calculating derivatives,
    // kSize should be 1, 3, 5 or 7
    const cv::Mat& CalSecDer(int kSize = 5, float linkEndBound = 0.01f,
                         float linkStartBound = 0.1f);
    const cv::Mat& CalFirDer(int kSize = 5, float linkEndBound = 0.01f,
                         float linkStartBound = 0.1f);
    const std::vector<CEdge>& Link(int shortRemoveBound = 3);

    // Get data pointers
    const cv::Mat& GetDer(){ return m_pDer1f; }
    const cv::Mat& GetLineIdx() { return m_pLabel1i; } // Edge index start from 1
    const cv::Mat& GetNextMap() { return m_pNext1i; }
    const cv::Mat& GetOrnt() { return m_pOrnt1f; }
    const std::vector<CEdge>& GetEdges() {return m_vEdge;}

    static const int IND_BG = 0xffffffff; // Background
    static const int IND_NMS = 0xfffffffe; // Non Maximal Suppress
    static const int IND_SR = 0xfffffffd; // and Short Remove

    static void Demo(const cv::Mat &img1u, bool isCartoon);

private:
    const cv::Mat &m_img1f; // Input image

    cv::Mat m_pDer1f;   // First or secondary derivatives. 32FC1
    cv::Mat m_pOrnt1f;  // Line orientation. 32FC1
    cv::Mat m_pLabel1i;  // Line index, 32SC1.
    cv::Mat m_pNext1i;   // Next point 8-direction index, [0, 1, ...,  7], 32SC1

    // Will be used for link process
    typedef std::pair<float, cv::Point> PntImp;
    std::vector<PntImp> m_StartPnt;
    std::vector<CEdge> m_vEdge;
    static bool linePointGreater (const PntImp& e1, const PntImp& e2 )
    {
        return e1.first > e2.first;
    }

    int m_h, m_w; // Image size
    int m_kSize; // Smooth kernel size: 1, 3, 5, 7
    float m_maxAngDif; // maximal allowed angle difference in a curve

    void NoneMaximalSuppress(float linkEndBound, float linkStartBound);
    void findEdge(cv::Point seed, CEdge& crtEdge, bool isBackWard);
    bool goNext(cv::Point &pnt, float& ornt, CEdge& crtEdge, int orntInd,
                bool isBackward);
    bool jumpNext(cv::Point &pnt, float& ornt, CEdge& crtEdge, int orntInd,
                  bool isBackward);

    /* Compute the eigenvalues and eigenvectors of the Hessian matrix given by
 dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
 their absolute values. */
    static void compute_eigenvals(double dfdrr, double dfdrc, double dfdcc,
                                  double eigval[2], double eigvec[2][2]);

    static inline float angle(float ornt1, float orn2);
    static inline void refreshOrnt(float& ornt, float& newOrnt);
};

typedef CmCurveEx::CEdge CmEdge;
typedef std::vector<CmEdge> CmEdges;

#endif // CMCURVEEX_HH
