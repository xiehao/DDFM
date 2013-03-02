#include <iostream>
#include <fstream>

#include "c_winner_take_all.hh"
#include "CmCurveEx.h"

using namespace xh;

const std::string workspace = "../../data/";

bool demo(void)
{
    /*!< source stereo images */
    cv::Mat L = cv::imread(workspace + "L.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat R = cv::imread(workspace + "R.png", CV_LOAD_IMAGE_COLOR);

    if (!L.data || !R.data)
    {
        std::cerr << "Error!!, Loading image failed!!" << std::endl;
        return false;
    }
#if 0
    cv::imshow("L", L);
    cv::imshow("R", R);
    cv::waitKey();
#endif
    cv::Mat_<cv::Vec3f> L_3f = cv::Mat_<cv::Vec3f>(L) * (1 / 255.f);
    cv::Mat_<cv::Vec3f> R_3f = cv::Mat_<cv::Vec3f>(R) * (1 / 255.f);

    /*!< edge masks */
    cv::Mat L_1b = cv::imread(workspace + "L-smooth.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat R_1b = cv::imread(workspace + "R-smooth.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat_<float> L_1f = cv::Mat_<float>(L_1b) * (1 / 255.f);
    cv::Mat_<float> R_1f = cv::Mat_<float>(R_1b) * (1 / 255.f);
    CmCurveEx curve_ex_L(L_1f);
    CmCurveEx curve_ex_R(R_1f);
    curve_ex_L.CalFirDer(5, 0.05f, 0.2f);
    curve_ex_R.CalFirDer(5, 0.05f, 0.2f);
    curve_ex_L.Link();
    curve_ex_R.Link();
    const CmEdges &edges_L = curve_ex_L.GetEdges();
    const CmEdges &edges_R = curve_ex_R.GetEdges();

    cv::Mat_<uchar> mask_L = cv::Mat_<uchar>::zeros(L.size());
    cv::Mat_<uchar> mask_R = cv::Mat_<uchar>::zeros(R.size());
    CmEdges::const_iterator e_cit = edges_L.begin();
//    CmEdges::const_iterator e_cit_end = edges_L.end();
    CmEdges::const_iterator e_cit_end = edges_L.begin() + 1;
    for (; e_cit != e_cit_end; ++e_cit)
    {
        const int step = 50;
        const std::vector<cv::Point> &points = e_cit->pnts;
        size_t n_points = points.size();
        for (size_t i = 0; i < n_points; i += step)
        {
            mask_L.at<uchar>(points[i]) = 255;
        }
    }
    e_cit = edges_R.begin();
    e_cit_end = edges_R.end();
    for (; e_cit != e_cit_end; ++e_cit)
    {
        const std::vector<cv::Point> &points = e_cit->pnts;
        std::vector<cv::Point>::const_iterator cit = points.begin();
        std::vector<cv::Point>::const_iterator cit_end = points.end();
        for (; cit != cit_end; ++cit)
        {
            mask_R.at<uchar>(*cit) = 255;
        }
    }

    cv::imshow("L-mask", mask_L);
    cv::imshow("R-mask", mask_R);
    cv::waitKey();
    cv::imwrite(workspace + "L-mask.png", mask_L);
    cv::imwrite(workspace + "R-mask.png", mask_R);

    /*!< do the WTA job */
    c_winner_take_all wta;
    wta.set_data(L_3f, R_3f, mask_L, mask_R);
    wta.solve(c_winner_take_all::E_INTENSITY, 50);

    /*!< read back results */
    const std::vector<s_match> &match_set = wta.get_match_set();
    std::vector<s_match>::const_iterator cit = match_set.begin();
    std::vector<s_match>::const_iterator cit_end = match_set.end();
    for (; cit != cit_end; ++cit)
    {
        const s_match &match = *cit;
        cv::circle(L, cv::Point(match.xl, match.y), 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(R, cv::Point(match.xr, match.y), 2, cv::Scalar(255, 255, 0), -1);
    }

    const std::vector<s_match> &unmatch_set = wta.get_unmatch_set();
    cit = unmatch_set.begin();
    cit_end = unmatch_set.end();
    for (; cit != cit_end; ++cit)
    {
        const s_match &match = *cit;
        cv::circle(L, cv::Point(match.xl, match.y), 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(R, cv::Point(match.xr, match.y), 2, cv::Scalar(0, 255, 255), -1);
    }

    /*!< show matches */
    cv::imshow("L", L);
    cv::imshow("R", R);
    cv::waitKey();
    cv::imwrite(workspace + "L-matches-wta.png", L);
    cv::imwrite(workspace + "R-matches-wta.png", R);

    return true;
}

int main(int argc, char *argv[])
{
    demo();
    std::cout << "Hello WTA!!" << std::endl;
    return 0;
}
