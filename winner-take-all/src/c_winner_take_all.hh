#ifndef C_WINNER_TAKE_ALL_HH
#define C_WINNER_TAKE_ALL_HH

#include <vector>
#include <opencv2/opencv.hpp>

namespace xh
{

struct s_match
{
    s_match(const float _xl, const float _xr, const float _y)
        : xl(_xl), xr(_xr), y(_y) { }
    float xl;
    float xr;
    float y;
};

class c_winner_take_all
{
public:
    enum e_type
    {
        E_INTENSITY,
        E_GRADIENT,
        E_GEODESIC
    };

    /*!< constructor */
    c_winner_take_all(void) { }
    c_winner_take_all(const cv::Mat_<cv::Vec3f> &_left,
                      const cv::Mat_<cv::Vec3f> &_right,
                      const cv::Mat_<uchar> &_mask_left = cv::Mat_<uchar>::zeros(1, 1),
                      const cv::Mat_<uchar> &_mask_right = cv::Mat_<uchar>::zeros(1, 1));

    /*!< set stereo images and their mask */
    bool set_data(const cv::Mat_<cv::Vec3f> &_left,
                  const cv::Mat_<cv::Vec3f> &_right,
                  const cv::Mat_<uchar> &_mask_left = cv::Mat_<uchar>::zeros(1, 1),
                  const cv::Mat_<uchar> &_mask_right = cv::Mat_<uchar>::zeros(1, 1));

    /*!< do the main work */
    bool solve(const e_type _type, const int _max_disparity,
               const int _n_size_windows = 31);

    /*!< read back results */
    const std::vector<s_match> &get_match_set(void) const
    {
        return m_match_set;
    }
    const cv::Mat_<float> &get_disparity_map(void) const
    {
        return m_disparity_map;
    }
    const std::vector<s_match> &get_unmatch_set(void) const
    {
        return m_unmatch_set;
    }
private:
#if 0
    /*!< find match for one pixel (no use) */
    bool find_match(const cv::Point &_point, cv::Point &_match);
#endif
    /*!< calculate geodesic distances for a local window */
    cv::Mat_<float> geodesic_distance(const cv::Mat_<cv::Vec3f> &_image,
                                      const cv::Point &_center,
                                      const cv::Vec4i &_offsets, const int _n_window_size);

    /*!< various cost functions */
    float cost_function(const cv::Point &_p_left, const cv::Point &_p_right,
                        const cv::Mat_<cv::Vec3f> &_image,
                        const e_type _type = E_INTENSITY,
                        const int _size_window = 31);
    float cost_function_intensity(const cv::Point &_p_left,
                                  const cv::Point &_p_right,
                                  const int _size_window = 31);
    float cost_function_gradient(const cv::Point &_p_left,
                                 const cv::Point &_p_right,
                                 const int _size_window = 31);
    float cost_function_geodesic(const cv::Point &_p_left,
                                 const cv::Point &_p_right,
                                 const cv::Mat_<cv::Vec3f> &_image,
                                 const int _size_window = 31);

    cv::Mat_<cv::Vec3f> m_left;
    cv::Mat_<cv::Vec3f> m_right;
    cv::Mat_<uchar> m_mask_left;
    cv::Mat_<uchar> m_mask_right;

    cv::Mat_<float> m_disparity_map;
    std::vector<s_match> m_match_set;
    std::vector<s_match> m_unmatch_set;
};

} // namespace xh

#endif // C_WINNER_TAKE_ALL_HH
