#ifndef S_GLOBAL_DATA_HH
#define S_GLOBAL_DATA_HH

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

namespace xh
{

class s_global_data
{
public:
    ~s_global_data(void) { }

    static s_global_data *get_instance(void)
    {
        if (!m_p_instance)
        {
            m_p_instance = new s_global_data();
        }
        return m_p_instance;
    }

    bool load(const std::string &_file_name);
    bool save(const std::string &_file_name);
    bool is_ready(void)
    {
        return m_is_ready;
    }

    /*!< global data */

    /*!< map from Index to Point for both iamges */
    std::vector<cv::Point> m_features_left;
    std::vector<cv::Point> m_features_right;
    /*!< map from Index to Match */
    std::vector<cv::Point> m_match_set;

    /*!< k nearest neighbor search for both images */
    cv::flann::Index m_knn_left;
    cv::flann::Index m_knn_right;
    cv::flann::Index m_knn_match_set;
    /*!< global data for feature points used in knn */
    cv::Mat_<float> m_features_left_cv;
    cv::Mat_<float> m_features_right_cv;
    cv::Mat_<float> m_match_set_cv;

    /*!< search matches from points on both iamges */
    std::vector<std::vector<int> > m_matches_left;
    std::map<int, std::vector<int> > m_matches_right;

    /*!< neighbor points of each point on both images */
    std::vector<std::vector<int> > m_neighbors_left;
    std::vector<std::vector<int> > m_neighbors_right;

#if 1
    /*!< neighbor set (N_P in the paper, currently not used)
     * NOTE that neighborhood relationship is irreversable */
    std::vector<cv::Point> m_neighbors_set_left;
    std::vector<cv::Point> m_neighbors_set_right;
#endif

private:
    s_global_data(void);
    s_global_data(const s_global_data &);

    static s_global_data *m_p_instance;
    bool m_is_ready;
};

} // namespace xh

#endif // S_GLOBAL_DATA_HH
