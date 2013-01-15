#ifndef S_ARGUMENTS_HH
#define S_ARGUMENTS_HH

#include <string>
#include <opencv2/opencv.hpp>

namespace xh
{

class s_arguments
{
public:
    ~s_arguments(void)
    {

    }

    static s_arguments *get_instance(void)
    {
        if (!m_p_instance)
        {
            m_p_instance = new s_arguments();
        }
        return m_p_instance;
    }

    bool load(const std::string &_file_name);
    bool save(const std::string &_file_name);
    bool is_ready(void)
    {
        return m_is_ready;
    }
    static bool generate_sample_configuration(void)
    {
        return s_arguments::get_instance()->save("config.ini");
    }

    /*!< global arguments */

    cv::Mat_<cv::Vec3f> m_left;
    cv::Mat_<cv::Vec3f> m_right;
    cv::Mat_<uchar> m_left_mask;
    cv::Mat_<uchar> m_right_mask;
    cv::Size m_image_size;
    int m_window_size;

    int m_max_disparity;

    /*!< coefficients of the four terms in the objective function */
    float m_lambda_appearance;
    float m_lambda_occlusion;
    float m_lambda_geometry;
    float m_lambda_coherence;

    float m_ita;
    float m_sigma_l;
    float m_sigma_a;

private:
    s_arguments(void);
    s_arguments(const s_arguments &);

    static s_arguments *m_p_instance;
    bool m_is_ready;
};

} // namespace xh

#endif // S_ARGUMENTS_HH
