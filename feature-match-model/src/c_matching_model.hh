#ifndef C_MATCHING_MODEL_HH
#define C_MATCHING_MODEL_HH

#include <tr1/memory>
#include <vector>
#include <opencv2/opencv.hpp>

namespace xh
{

class s_arguments;
class s_global_data;

/*!< class defining a pair of matching points in stereo images */
class s_pair
{
public:
    s_pair(void)
        : m_xl(0), m_xr(0), m_y(0) { }
    s_pair(const int _xl, const int _xr, const int _y)
        : m_xl(_xl), m_xr(_xr), m_y(_y) { }
    int m_xl;   /*!< x coordinate in the left image */
    int m_xr;   /*!< x coordinate in the right image */
    int m_y;    /*!< y coordinate in the both image */
};

/*!< class of preparation for feature matching in stereo images */
class c_matching_model
{
public:
    typedef std::vector<cv::Point> t_feature_set;
    typedef std::vector<cv::Point> t_match_set;
    typedef std::vector<std::map<int, float> > t_coefficient_set;

    /*!< default constructor */
    c_matching_model(void);

    /*!< build the objective function from the feature matching problem */
    bool build(void);

    /*!< output results into a file */
    bool save(const std::string &_file_name);

    /*!< demonstration of class usage */
    static void demo(void);

private:
    /*!< prepare data */
    bool prepare_data(void);

    /*!< construct the appearance term (similarity in between pair element) */
    bool prepare_appearance_term(void);

    /*!< construct the occlusion term (penalty for unmatched pairs) */
    bool prepare_occlusion_term(void);

    /*!< construct the geometric term (geometric compatibility between pairs) */
    bool prepare_geometry_term(void);

    /*!< construct the coherence term (shape preservation) */
    bool prepare_coherence_term(void);

    /*!< initialize the global data */
    bool initiate_data(void);
    /*!< cost function measuring a matching pair  */
    float cost_function(const cv::Point &_p_left, const cv::Point &_p_right);
    /*!< build kd tree for further search */
    bool build_knn(const t_feature_set &_points, cv::Mat_<float> &_features,
                   cv::flann::Index &_knn);
    /*!< build neighbor set for each point on both images */
    bool build_neighbors(const t_feature_set &_points, cv::flann::Index &_knn,
                         std::vector<std::vector<int> > &_neighbors,
                         const int _k = 3);
    /*!< build neighbor set system for both images (without duplications) */
    size_t build_neighbors_set(void);
    /*!< get index of an element from set */
    int get_index(cv::flann::Index &_knn, const cv::Point &_query);
    void get_knn_results(const cv::Point &_query, cv::flann::Index &_knn,
                         std::vector<int> &_indices, const int _k);
    /*!< check if point is one of another point's neighbors */
    bool in_neighbor(const int _i_point, const std::vector<int> &_neighbors)
    {
        return std::find(_neighbors.begin(), _neighbors.end(), _i_point) !=
                _neighbors.end();
    }
    /*!< check if two matches are in the same neighbor system */
    bool in_neighbor_system(std::vector<cv::Point>::const_iterator _cit_a,
                            std::vector<cv::Point>::const_iterator _cit_b);
    /*!< calculate geometric consistency between two matches */
    float geometric_consistency(std::vector<cv::Point>::const_iterator _cit_a,
                                std::vector<cv::Point>::const_iterator _cit_b);

    /*!< data members */

    /*!< smart pointer to global arguments */
    std::tr1::shared_ptr<s_arguments> m_p_arguments;
    cv::Mat_<uchar> m_new_right_mask;
    /*!< smart pointer to the global data */
    std::tr1::shared_ptr<s_global_data> m_p_global_data;

    /*!< coefficients of the first order terms in the objective function */
    std::vector<float> m_coefficients_unary;
    /*!< coefficients of the second order terms in the objective function */
    t_coefficient_set m_coefficients_binary;
};

} // namespace xh

#endif // C_MATCHING_MODEL_HH
