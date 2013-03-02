#include "c_winner_take_all.hh"

namespace xh
{

c_winner_take_all::c_winner_take_all(const cv::Mat_<cv::Vec3f> &_left,
                                     const cv::Mat_<cv::Vec3f> &_right,
                                     const cv::Mat_<uchar> &_mask_left,
                                     const cv::Mat_<uchar> &_mask_right)
    : m_left(_left), m_right(_right), m_mask_left(_mask_left),
      m_mask_right(_mask_right)
{
    assert(m_left.size() == m_right.size());
    if (1 == m_mask_left.rows)
    {
        m_mask_left = cv::Mat_<uchar>::ones(m_left.size());
    }
    if (1 == m_mask_right.rows)
    {
        m_mask_right = cv::Mat_<uchar>::ones(m_right.size());
    }
    assert(m_mask_left.size() == m_left.size());
    assert(m_mask_right.size() == m_right.size());

    m_disparity_map = cv::Mat_<float>::zeros(m_left.size());

    return;
}

bool c_winner_take_all::set_data(const cv::Mat_<cv::Vec3f> &_left,
                                 const cv::Mat_<cv::Vec3f> &_right,
                                 const cv::Mat_<uchar> &_mask_left,
                                 const cv::Mat_<uchar> &_mask_right)
{
    assert(_left.size() == _right.size());
    m_left = _left;
    m_right = _right;
    m_mask_left = _mask_left;
    m_mask_right = _mask_right;
    if (1 == _mask_left.rows)
    {
        m_mask_left = cv::Mat_<uchar>::ones(_left.size());
    }
    if (1 == _mask_right.rows)
    {
        m_mask_right = cv::Mat_<uchar>::ones(_right.size());
    }
    assert(m_mask_left.size() == _left.size());
    assert(m_mask_right.size() == _right.size());

    m_disparity_map = cv::Mat_<float>::zeros(m_left.size());

    return true;
}

bool c_winner_take_all::solve(const e_type _type, const int _max_disparity,
                              const int _size_windows)
{
    const float max_cost = 1e+5;
    int offset = _max_disparity;
    m_match_set.clear();
    m_unmatch_set.clear();

    int width = m_left.cols;
    int height = m_left.rows;

    for (int i_row = 0; i_row < height; ++i_row)
    {
        uchar *p_left = m_mask_left.ptr<uchar>(i_row);
        uchar *p_right = m_mask_right.ptr<uchar>(i_row);
        float *p_map = m_disparity_map.ptr<float>(i_row);
        for (int i_col = 0; i_col < width; ++i_col)
        {
            if (p_left[i_col])
            {
                /*!< for each query point, search its match via WTA */
                float min_cost = max_cost;
                int i_min = -1;
                int left = std::max(0, i_col - offset);
                int right = std::min(width, i_col + offset);
                for (int i = left; i < right; ++i)
                {
                    if (p_right[i])
                    {
                        float cost = cost_function(cv::Point(i_col, i_row),
                                                   cv::Point(i, i_row),
                                                   m_left, _type, _size_windows);
                        if (min_cost > cost)
                        {
                            min_cost = cost;
                            i_min = i;
                        }
                    }
                }

                /*!< cross checking to eleminate occlusion errors */
                if (-1 != i_min)
                {
                    int j_min = -1;
                    min_cost = max_cost;
                    for (int j = left; j < right; ++j)
                    {
                        float cost = cost_function(cv::Point(j, i_row),
                                                   cv::Point(i_min, i_row),
                                                   m_right, _type, _size_windows);
                        if (min_cost > cost)
                        {
                            min_cost = cost;
                            j_min = j;
                        }
                    }

                    if (std::abs(j_min - i_col) < 1)
                    {
                        p_map[i_min] = i_min - i_col;
                        m_match_set.push_back(s_match(i_col, i_min, i_row));
                    }
                    else
                    {
//                        m_unmatch_set.push_back(s_match(i_col, i_min, i_row));
                    }
                }
            }
        }
    }

    return m_match_set.size();
}
#if 0
bool c_winner_take_all::find_match(const cv::Point &_point, cv::Point &_match)
{
    _match = cv::Point(-1, -1);
    return true;
}
#endif
cv::Mat_<float>
c_winner_take_all::geodesic_distance(const cv::Mat_<cv::Vec3f> &_image,
                                     const cv::Point &_center,
                                     const cv::Vec4i &_offsets,
                                     const int _n_window_size)
{
    cv::Mat_<float> geodesic_map = cv::Mat_<float>::ones(_n_window_size,
                                                         _n_window_size) * 1e8;

    int radius = _n_window_size / 2;
    geodesic_map.at<float>(radius, radius) = 0;

    int d_top = _offsets.val[0];
    int d_buttom = _offsets.val[1];
    int d_left = _offsets.val[2];
    int d_right = _offsets.val[3];

    const int n_iter = 3;
    for (int i_iter = 0; i_iter < n_iter; ++i_iter)
    {
        /*!< forward pass */
        for (int d_row = 1 - d_top; d_row <= d_buttom; ++d_row)
        {
            const cv::Vec3f *p = _image.ptr<cv::Vec3f>(d_row + _center.y);
            const cv::Vec3f *p_prev = _image.ptr<cv::Vec3f>(d_row - 1 + _center.y);
            float *q = geodesic_map.ptr<float>(d_row + radius);
            float *q_prev = geodesic_map.ptr<float>(d_row - 1 + radius);
            for (int d_col = 1 - d_left; d_col < d_right; ++d_col)
            {
                cv::Vec3f current = p[d_col + _center.x];
                std::vector<float> costs(5);
                costs[0] = q_prev[d_col - 1 + radius] +
                        cv::norm(p_prev[d_col - 1 + _center.x], current, cv::NORM_L2);
                costs[1] = q_prev[d_col + radius] +
                        cv::norm(p_prev[d_col + _center.x], current, cv::NORM_L2);
                costs[2] = q_prev[d_col + 1 + radius] +
                        cv::norm(p_prev[d_col + 1 + _center.x], current, cv::NORM_L2);
                costs[3] = q[d_col - 1 + radius] +
                        cv::norm(p[d_col - 1 + _center.x], current, cv::NORM_L2);
                costs[4] = q[d_col + radius];
                float min_cost = *std::min_element(costs.begin(), costs.end());
                q[d_col + radius] = min_cost;
            }
        }

        /*!< backward pass */
        for (int d_row = 1 - d_buttom; d_row >= d_top; --d_row)
        {
            const cv::Vec3f *p = _image.ptr<cv::Vec3f>(d_row + _center.y);
            const cv::Vec3f *p_post = _image.ptr<cv::Vec3f>(d_row + 1 + _center.y);
            float *q = geodesic_map.ptr<float>(d_row + radius);
            float *q_post = geodesic_map.ptr<float>(d_row + 1 + radius);
            for (int d_col = 1 - d_right; d_col > d_left; --d_col)
            {
                cv::Vec3f current = p[d_col + _center.x];
                std::vector<float> costs(5);
                costs[0] = q_post[d_col - 1 + radius] +
                        cv::norm(p_post[d_col - 1 + _center.x], current, cv::NORM_L2);
                costs[1] = q_post[d_col + radius] +
                        cv::norm(p_post[d_col + _center.x], current, cv::NORM_L2);
                costs[2] = q_post[d_col + 1 + radius] +
                        cv::norm(p_post[d_col + 1 + _center.x], current, cv::NORM_L2);
                costs[3] = q[d_col + 1 + radius] +
                        cv::norm(p[d_col + 1 + _center.x], current, cv::NORM_L2);
                costs[4] = q[d_col + radius];
                float min_cost = *std::min_element(costs.begin(), costs.end());
                q[d_col + radius] = min_cost;
            }
        }
    }

    return geodesic_map;
}

float c_winner_take_all::cost_function(const cv::Point &_p_left,
                                       const cv::Point &_p_right, const cv::Mat_<cv::Vec3f> &_image,
                                       const e_type _type,
                                       const int _size_window)
{
    float cost = 0;
    switch (_type)
    {
    case E_INTENSITY:
        cost = cost_function_intensity(_p_left, _p_right, _size_window);
        break;
    case E_GRADIENT:
        cost = cost_function_gradient(_p_left, _p_right, _size_window);
        break;
    case E_GEODESIC:
        cost = cost_function_geodesic(_p_left, _p_right, _image, _size_window);
        break;
    default:
        break;
    }
    return cost;
}

float c_winner_take_all::cost_function_intensity(const cv::Point &_p_left,
                                                 const cv::Point &_p_right,
                                                 const int _size_window)
{
    int radius = _size_window / 2;
    int right = m_left.cols - 1;
    int down = m_left.rows - 1;

    int d_top = std::min(std::min(_p_left.y - 0, _p_right.y - 0), radius);
    int d_buttom = std::min(std::min(down - _p_left.y, down - _p_right.y), radius);
    int d_left = std::min(std::min(_p_left.x - 0, _p_right.x - 0), radius);
    int d_right = std::min(std::min(right - _p_left.x, right - _p_right.x), radius);

    float cost = 0.f;
    for (int d_row = -d_top; d_row <= d_buttom; ++d_row)
    {
        cv::Vec3f *p_left = m_left.ptr<cv::Vec3f>(d_row + _p_left.y);
        cv::Vec3f *p_right = m_right.ptr<cv::Vec3f>(d_row + _p_right.y);
        for (int d_col = -d_left; d_col <= d_right; ++d_col)
        {
            cost += cv::norm(p_left[d_col + _p_left.x],
                    p_right[d_col + _p_right.x], cv::NORM_L2);
        }
    }

    return cost;
}

float c_winner_take_all::cost_function_gradient(const cv::Point &_p_left,
                                                const cv::Point &_p_right,
                                                const int _size_window)
{
    return 0;
}

float c_winner_take_all::cost_function_geodesic(const cv::Point &_p_left,
                                                const cv::Point &_p_right,
                                                const cv::Mat_<cv::Vec3f> &_image,
                                                const int _window_size)
{
    int radius = _window_size / 2;
    int right = m_left.cols - 1;
    int down = m_left.rows - 1;

    int d_top = std::min(std::min(_p_left.y - 0, _p_right.y - 0), radius);
    int d_buttom = std::min(std::min(down - _p_left.y, down - _p_right.y), radius);
    int d_left = std::min(std::min(_p_left.x - 0, _p_right.x - 0), radius);
    int d_right = std::min(std::min(right - _p_left.x, right - _p_right.x), radius);

    const cv::Vec4i offsets(d_top, d_buttom, d_left, d_right);
    cv::Mat_<float> geodesic = geodesic_distance(_image, _p_left, offsets,
                                                 _window_size);

    float cost = 0.f;
    const float gamma = 2;
    for (int d_row = -d_top; d_row <= d_buttom; ++d_row)
    {
        cv::Vec3f *p_left = m_left.ptr<cv::Vec3f>(d_row + _p_left.y);
        cv::Vec3f *p_right = m_right.ptr<cv::Vec3f>(d_row + _p_right.y);
        float *p = geodesic.ptr<float>(d_row + radius);
        for (int d_col = -d_left; d_col <= d_right; ++d_col)
        {
            float weight = std::exp(-p[d_col + radius] / gamma);
            cost += weight * cv::norm(p_left[d_col + _p_left.x],
                    p_right[d_col + _p_right.x], cv::NORM_L2);
        }
    }

    return cost;
}

} // namespace xh
