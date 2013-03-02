#include "c_matching_model.hh"

#include <fstream>
#include <set>

#include "s_arguments.hh"
#include "s_global_data.hh"

namespace xh
{

const std::string workspace = "../../data/";

c_matching_model::c_matching_model(void)
    : m_p_arguments(s_arguments::get_instance()),
      m_new_right_mask(cv::Mat_<uchar>::zeros(s_arguments::get_instance()->m_image_size)),
      m_p_global_data(s_global_data::get_instance())
{
    assert(m_p_arguments->is_ready());
    assert(initiate_data());
}

bool c_matching_model::build(void)
{
    std::cout << "Preparing data..." << std::endl;
    prepare_data();

    std::cout << "Preparing appearance term..." << std::endl;
    prepare_appearance_term();
    std::cout << "Preparing occlusion term..." << std::endl;
    prepare_occlusion_term();
    std::cout << "Preparing geometry term..." << std::endl;
    prepare_geometry_term();
    std::cout << "Preparing coherence term..." << std::endl;
    prepare_coherence_term();
    std::cout << "Building objective done!!" << std::endl;

    return true;
}

bool c_matching_model::save(const std::string &_file_name)
{
    std::ofstream file(_file_name.c_str());
    file.clear();

    file << "c Graph Matching problem for correspondences on stero images"
         << std::endl;

    /*!< print general information for the whole problem */
    int n_left = m_p_global_data->m_features_left.size();
    int n_right = m_p_global_data->m_features_right.size();
    int n_assignments = m_coefficients_unary.size();
    int n_edges = 0;
    t_coefficient_set::const_iterator cit_b = m_coefficients_binary.begin();
    t_coefficient_set::const_iterator cit_b_end = m_coefficients_binary.end();
    for (; cit_b != cit_b_end; ++cit_b)
    {
        n_edges += cit_b->size();
    }
    file << "p " << n_left << ' ' << n_right << ' ' << n_assignments << ' '
         << n_edges << std::endl;

    /*!< print assignments (matches) */
    const std::vector<cv::Point> &match_set = m_p_global_data->m_match_set;
    std::vector<float>::const_iterator cit_u = m_coefficients_unary.begin();
    std::vector<float>::const_iterator cit_u_end = m_coefficients_unary.end();
    for (size_t i = 0; cit_u != cit_u_end; ++cit_u, ++i)
    {
        const cv::Point &match = match_set[i];
        file << "a " << i << ' ' << match.x << ' ' << match.y << ' ' << *cit_u
             << std::endl;
    }

    /*!< print edges (pairs between matches) */
    cit_b = m_coefficients_binary.begin();
    cit_b_end = m_coefficients_binary.end();
    for (size_t a = 0; cit_b != cit_b_end; ++cit_b, ++a)
    {
        const std::map<int, float> &binary = *cit_b;
        if (binary.size())
        {
            std::map<int, float>::const_iterator cit = binary.begin();
            std::map<int, float>::const_iterator cit_end = binary.end();
            for (; cit != cit_end; ++cit)
            {
                file << "e " << a << ' ' << cit->first << ' ' << cit->second
                     << std::endl;
            }
        }
    }
#if 1
    /*!< OPTIONAL: print neighbors */
    const std::vector<std::vector<int> > &left_neighbors =
            m_p_global_data->m_neighbors_left;
    std::vector<std::vector<int> >::const_iterator cit_n = left_neighbors.begin();
    std::vector<std::vector<int> >::const_iterator cit_n_end = left_neighbors.end();
    for (size_t p = 0; cit_n != cit_n_end; ++cit_n, ++p)
    {
        std::vector<int>::const_iterator cit = cit_n->begin();
        std::vector<int>::const_iterator cit_end = cit_n->end();
        for (; cit != cit_end; ++cit)
        {
            file << "n0 " << p << ' ' << *cit << std::endl;
        }
    }
    const std::vector<std::vector<int> > &right_neighbors =
            m_p_global_data->m_neighbors_right;
    cit_n = right_neighbors.begin();
    cit_n_end = right_neighbors.end();
    for (size_t p = 0; cit_n != cit_n_end; ++cit_n, ++p)
    {
        std::vector<int>::const_iterator cit = cit_n->begin();
        std::vector<int>::const_iterator cit_end = cit_n->end();
        for (; cit != cit_end; ++cit)
        {
            file << "n1 " << p << ' ' << *cit << std::endl;
        }
    }
#endif
    file.close();
    return true;
}

void c_matching_model::demo(void)
{
    std::string file_name = "config.ini";
    s_arguments::get_instance()->load(file_name);
    c_matching_model em;
    em.build();
    em.save(workspace + "DATA.TXT");

    return;
}

bool c_matching_model::prepare_data(void)
{
    int width = m_p_arguments->m_image_size.width;
    int height = m_p_arguments->m_image_size.height;

    const cv::Mat_<uchar> &left_mask = m_p_arguments->m_left_mask;
    cv::Mat_<uchar> &right_mask = m_new_right_mask;
    const int max_disparity = m_p_arguments->m_max_disparity;

    /*!< construct the left features and right mask */
    t_feature_set &features_left = m_p_global_data->m_features_left;
    for (int i_row = 0; i_row < height; ++i_row)
    {
        const uchar *p_left_mask = left_mask.ptr<uchar>(i_row);
        uchar *p_right_mask = right_mask.ptr<uchar>(i_row);
        for (int i_col = 0; i_col < width; ++i_col)
        {
            if (p_left_mask[i_col])
            {
                cv::Point point_left(i_col, i_row);
                features_left.push_back(point_left);

                for (int k = 0; k <= max_disparity; ++k)
                {
                    int i = i_col - k;
                    if (i >= 0)
                    {
                        p_right_mask[i] = 127;
                    }
                }
            }
        }
    }

    cv::imshow("right mask", right_mask);
    cv::waitKey();

    cv::flann::Index &knn_left = m_p_global_data->m_knn_left;
    cv::Mat_<float> &features_left_cv = m_p_global_data->m_features_left_cv;
    build_knn(features_left, features_left_cv, knn_left);
    build_neighbors(features_left, knn_left,
                    m_p_global_data->m_neighbors_left);
    save_features(workspace + "left-features.txt", features_left);

    /*!< construct the right features from right mask */
    t_feature_set &features_right = m_p_global_data->m_features_right;
    for (int i_row = 0; i_row < height; ++i_row)
    {
        const uchar *p = right_mask.ptr<uchar>(i_row);
        for (int i_col = 0; i_col < width; ++i_col)
        {
            if (p[i_col])
            {
                cv::Point point_right(i_col, i_row);
                features_right.push_back(point_right);
            }
        }
    }

    cv::flann::Index &knn_right = m_p_global_data->m_knn_right;
    cv::Mat_<float> &feature_right_cv = m_p_global_data->m_features_right_cv;
    build_knn(features_right, feature_right_cv, knn_right);
    build_neighbors(features_right, knn_right,
                    m_p_global_data->m_neighbors_right);
    save_features(workspace + "right-features.txt", features_right);


    /*!< construct matches from both images */
    std::vector<std::vector<int> > &map_left = m_p_global_data->m_matches_left;
    std::map<int, std::vector<int> > &map_right = m_p_global_data->m_matches_right;
    t_match_set &match_set = m_p_global_data->m_match_set;
    t_feature_set::const_iterator cit_beg = features_left.begin();
    t_feature_set::const_iterator cit_end = features_left.end();
    t_feature_set::const_iterator cit = cit_beg;
    for (int i_left = 0; cit != cit_end; ++cit, ++i_left)
    {
        int i_row = cit->y;
        int i_col = cit->x;
        int i_col_beg = std::max(0, i_col - max_disparity);
        std::vector<int> &match_left = map_left[i_left];
        match_left.reserve(i_col - i_col_beg + 1);
        const uchar *p_mask = right_mask.ptr<uchar>(i_row);
        for (; i_col >= i_col_beg; --i_col)
        {
            if (p_mask[i_col])
            {
                cv::Point point_right(i_col, i_row);
                int i_right = get_index(knn_right, point_right);

                match_left.push_back(i_right);
                map_right[i_right].push_back(i_left);
                match_set.push_back(cv::Point(i_left, i_right));
            }
        }
    }
    m_coefficients_unary.reserve(match_set.size());
    m_coefficients_binary.resize(match_set.size());

    cv::Mat_<float> &match_set_cv = m_p_global_data->m_match_set_cv;
    build_knn(match_set, match_set_cv, m_p_global_data->m_knn_match_set);

    return true;
}

bool c_matching_model::save_features(const std::string &_file_name,
                                     const t_feature_set &_features)
{
    std::ofstream file(_file_name.c_str());
    file.clear();

    t_feature_set::const_iterator cit = _features.begin();
    t_feature_set::const_iterator cit_end = _features.end();
    for (int i = 0; cit != cit_end; ++cit, ++i)
    {
        file << i << '\t' << cit->x << '\t' << cit->y << std::endl;
    }

    file.close();
    return true;
}

bool c_matching_model::prepare_appearance_term(void)
{
    const float lambda = m_p_arguments->m_lambda_appearance;
    const t_match_set &matches = m_p_global_data->m_match_set;
    const t_feature_set &left = m_p_global_data->m_features_left;
    const t_feature_set &right = m_p_global_data->m_features_right;
    t_match_set::const_iterator cit = matches.begin();
    t_match_set::const_iterator cit_end = matches.end();
    for (; cit != cit_end; ++cit)
    {
        const int i_left = cit->x;
        const int i_right = cit->y;

        const float cost = cost_function(left[i_left], right[i_right]);
        m_coefficients_unary.push_back(cost * lambda);
    }

    return true;
}

bool c_matching_model::prepare_occlusion_term(void)
{
    /*!< the left image has less features */
    const int n_features = cv::countNonZero(m_p_arguments->m_left_mask);
    if (n_features < 1)
    {
        std::cerr << "Number of features invalid!!" << std::endl;
        return false;
    }
    const float lambda = m_p_arguments->m_lambda_occlusion;
    const float coeffecient = -1.f / n_features;
    std::vector<float>::iterator it = m_coefficients_unary.begin();
    std::vector<float>::iterator it_end = m_coefficients_unary.end();
    for (; it != it_end; ++it)
    {
        *it += coeffecient * lambda;
    }

    return true;
}

bool c_matching_model::prepare_geometry_term(void)
{
    /*!< for every two matching pairs, analysis geometric consistency */
    const float lambda = m_p_arguments->m_lambda_geometry;
    const t_match_set &match_set = m_p_global_data->m_match_set;
    t_coefficient_set::iterator it = m_coefficients_binary.begin() + 1;
    t_coefficient_set::iterator it_end = m_coefficients_binary.end();
    t_match_set::const_iterator cit_beg = match_set.begin();
    t_match_set::const_iterator cit_end = match_set.end();
    t_match_set::const_iterator cit_a = cit_beg + 1;
    for (; cit_a != cit_end && it != it_end; ++cit_a, ++it)
    {
        t_match_set::const_iterator cit_b = cit_beg;
        for (; cit_b != cit_a; ++cit_b)
        {
            if (in_neighbor_system(cit_a, cit_b))
            {
                float theta = geometric_consistency(cit_a, cit_b);
                (*it)[cit_b - cit_beg] = theta * lambda;
            }
        }
    }
    return true;
}

bool c_matching_model::prepare_coherence_term(void)
{
    /*!< for every two neighbors, analysis coherence consistency */
    size_t n_neighbor_set = build_neighbors_set();
    if (n_neighbor_set < 2)
    {
        std::cerr << "Error! Too few elements in neighbors set N_p..."
                  << std::endl;
        return false;
    }

    const std::vector<cv::Point> &match_set = m_p_global_data->m_match_set;
    cv::flann::Index &knn = m_p_global_data->m_knn_match_set;
    const float lambda = m_p_arguments->m_lambda_coherence;

    const float coefficient_u = lambda / n_neighbor_set;
    const float coefficient_b = -2 * lambda / n_neighbor_set;

    /*!< left features */
    std::cout << "\tfor left features..." << std::endl;
    size_t n_points = m_p_global_data->m_features_left.size();
    const std::vector<std::vector<int> > &left_matches =
            m_p_global_data->m_matches_left;

    std::vector<cv::Point>::const_iterator cit =
            m_p_global_data->m_neighbors_set_left.begin();
    std::vector<cv::Point>::const_iterator cit_end =
            m_p_global_data->m_neighbors_set_left.end();
    for (; cit != cit_end; ++cit)
    {
        int p = cit->x;
        int q = cit->y;
        std::vector<int> indices_p, indices_q; /*!< indices of matches */
        indices_p.reserve(left_matches[p].size());
        indices_q.reserve(left_matches[q].size());

        /*!< unary coefficients */
        std::vector<int>::const_iterator cit_m = left_matches[p].begin();
        std::vector<int>::const_iterator cit_m_end = left_matches[p].end();
        for (; cit_m != cit_m_end; ++cit_m)
        {
            int index = get_index(knn, cv::Point(p, *cit_m));
            m_coefficients_unary[index] += coefficient_u;
            indices_p.push_back(index);
        }
        cit_m = left_matches[q].begin();
        cit_m_end = left_matches[q].end();
        for (; cit_m != cit_m_end; ++cit_m)
        {
            int index = get_index(knn, cv::Point(q, *cit_m));
            m_coefficients_unary[index] += coefficient_u;
            indices_q.push_back(index);
        }

        /*!< binary coefficients */
        std::vector<int>::const_iterator cit_p = indices_p.begin();
        std::vector<int>::const_iterator cit_p_end = indices_p.end();
        for (; cit_p != cit_p_end; ++cit_p)
        {
            int i_p = *cit_p;
            int i_p_match = match_set[i_p].y;
            std::vector<int>::const_iterator cit_q = indices_q.begin();
            std::vector<int>::const_iterator cit_q_end = indices_q.end();
            for (; cit_q != cit_q_end; ++cit_q)
            {
                int i_q = *cit_q;
                int i_q_match = match_set[i_q].y;
                if (i_q_match == i_p_match)
                {
                    continue;
                }
                if (i_p < i_q)
                {
                    m_coefficients_binary[i_q][i_p] += coefficient_b;
                }
                else
                {
                    m_coefficients_binary[i_p][i_q] += coefficient_b;
                }
            }
        }
    }

    /*!< right features */
    std::cout << "\tfor right features..." << std::endl;
    n_points = m_p_global_data->m_features_right.size();
    const std::map<int, std::vector<int> > &right_matches =
            m_p_global_data->m_matches_right;

    cit = m_p_global_data->m_neighbors_set_right.begin();
    cit_end = m_p_global_data->m_neighbors_set_right.end();
    for (; cit != cit_end; ++cit)
    {
        int p = cit->x;
        int q = cit->y;
        std::vector<int> indices_p, indices_q; /*!< indices of matches */
        const std::vector<int> &matches_p = right_matches.find(p)->second;
        const std::vector<int> &matches_q = right_matches.find(q)->second;
        indices_p.reserve(matches_p.size());
        indices_q.reserve(matches_q.size());

        /*!< unary coefficients */
        std::vector<int>::const_iterator cit_m = matches_p.begin();
        std::vector<int>::const_iterator cit_m_end = matches_p.end();
        for (; cit_m != cit_m_end; ++cit_m)
        {
            int index = get_index(knn, cv::Point(*cit_m, p));
            m_coefficients_unary[index] += coefficient_u;
            indices_p.push_back(index);
        }
        cit_m = matches_q.begin();
        cit_m_end = matches_q.end();
        for (; cit_m != cit_m_end; ++cit_m)
        {
            int index = get_index(knn, cv::Point(*cit_m, q));
            m_coefficients_unary[index] += coefficient_u;
            indices_q.push_back(index);
        }

        /*!< binary coefficients */
        std::vector<int>::const_iterator cit_p = indices_p.begin();
        std::vector<int>::const_iterator cit_p_end = indices_p.end();
        for (; cit_p != cit_p_end; ++cit_p)
        {
            int i_p = *cit_p;
            int i_p_match = match_set[i_p].x;
            std::vector<int>::const_iterator cit_q = indices_q.begin();
            std::vector<int>::const_iterator cit_q_end = indices_q.end();
            for (; cit_q != cit_q_end; ++cit_q)
            {
                int i_q = *cit_q;
                int i_q_match = match_set[i_q].x;
                if (i_q_match == i_p_match)
                {
                    continue;
                }
                if (i_p < i_q)
                {
                    m_coefficients_binary[i_q][i_p] += coefficient_b;
                }
                else
                {
                    m_coefficients_binary[i_p][i_q] += coefficient_b;
                }
            }
        }
    }
    return true;
}

bool c_matching_model::initiate_data(void)
{
    int n_left_features = cv::countNonZero(m_p_arguments->m_left_mask);
    int n_reserved = n_left_features * m_p_arguments->m_max_disparity;

    m_p_global_data->m_features_left.reserve(n_left_features);
    m_p_global_data->m_match_set.reserve(n_reserved);
    m_p_global_data->m_matches_left.resize(n_left_features);

    return true;
}

float c_matching_model::cost_function(const cv::Point &_p_left,
                                      const cv::Point &_p_right)
{
    int radius = m_p_arguments->m_window_size / 2;
    int right = m_p_arguments->m_image_size.width - 1;
    int down = m_p_arguments->m_image_size.height - 1;

    const cv::Mat_<cv::Vec3f> &left_image = m_p_arguments->m_left;
    const cv::Mat_<cv::Vec3f> &right_image = m_p_arguments->m_right;

    int d_top = std::min(std::min(_p_left.y - 0, _p_right.y - 0), radius);
    int d_buttom = std::min(std::min(down - _p_left.y, down - _p_right.y), radius);
    int d_left = std::min(std::min(_p_left.x - 0, _p_right.x - 0), radius);
    int d_right = std::min(std::min(right - _p_left.x, right - _p_right.x), radius);

    float cost = 0.f;
    for (int d_row = -d_top; d_row <= d_buttom; ++d_row)
    {
        const cv::Vec3f *p_left = left_image.ptr<cv::Vec3f>(d_row + _p_left.y);
        const cv::Vec3f *p_right = right_image.ptr<cv::Vec3f>(d_row + _p_right.y);
        for (int d_col = -d_left; d_col <= d_right; ++d_col)
        {
            cost += cv::norm(p_left[d_col + _p_left.x],
                    p_right[d_col + _p_right.x], cv::NORM_L2);
        }
    }

    return 0;
}

bool c_matching_model::build_knn(const t_feature_set &_points,
                                 cv::Mat_<float> &_features,
                                 cv::flann::Index &_knn)
{
    size_t n_points = _points.size();
    _features = cv::Mat_<float>::zeros(n_points, 2);
    t_feature_set::const_iterator cit = _points.begin();
    t_feature_set::const_iterator cit_end = _points.end();
    for (size_t i = 0; cit != cit_end; ++i, ++cit)
    {
        float *p = _features.ptr<float>(i);
        p[0] = cit->x;
        p[1] = cit->y;
    }

#if 0 // for debug
    std::cout << "Points for building knn:" << std::endl;
    std::cout << features << std::endl;
#endif

    _knn.build(_features, cv::flann::KDTreeIndexParams());

    return true;
}

bool c_matching_model::build_neighbors(const t_feature_set &_points,
                                       cv::flann::Index &_knn,
                                       std::vector<std::vector<int> > &_neighbors,
                                       const int _k)
{
    size_t n_points = _points.size();
    _neighbors.resize(n_points);
    t_feature_set::const_iterator cit = _points.begin();
    t_feature_set::const_iterator cit_end = _points.end();
    for (size_t i = 0; cit != cit_end; ++cit, ++i)
    {
        get_knn_results(*cit, _knn, _neighbors[i], _k);
#if 1 // for debug
        std::cout << "Neighbors of point " << get_index(_knn, *cit)
                  << ' ' << *cit << std::endl;
        std::vector<int>::const_iterator cit_n = _neighbors[i].begin();
        std::vector<int>::const_iterator cit_n_end = _neighbors[i].end();
        for (; cit_n != cit_n_end; ++cit_n)
        {
            std::cout << *cit_n << ' ' << _points[*cit_n] << '\t';
        }
        std::cout << std::endl;
#endif
    }

    return true;
}

size_t c_matching_model::build_neighbors_set(void)
{
    /*!< left features */
    std::cout << "\tfor left features..." << std::endl;
    size_t n_points = m_p_global_data->m_features_left.size();
    const std::vector<std::vector<int> > &left_neighbors =
            m_p_global_data->m_neighbors_left;
    std::vector<cv::Point> &left_set = m_p_global_data->m_neighbors_set_left;
    for (size_t p = 0; p < n_points; ++p)
    {
        for (size_t q = 0; q < p; ++q)
        {
            const std::vector<int> &neighbors_p = left_neighbors[p];
            const std::vector<int> &neighbors_q = left_neighbors[q];
            if (in_neighbor(q, neighbors_p) || in_neighbor(p, neighbors_q))
            {
                left_set.push_back(cv::Point(p, q));
            }
        }
    }

    /*!< right features */
    std::cout << "\tfor right features..." << std::endl;
    n_points = m_p_global_data->m_features_right.size();
    const std::vector<std::vector<int> > &right_neighbors =
            m_p_global_data->m_neighbors_right;
    std::vector<cv::Point> &right_set = m_p_global_data->m_neighbors_set_right;
    for (size_t p = 0; p < n_points; ++p)
    {
        for (size_t q = 0; q < p; ++q)
        {
            const std::vector<int> &neighbors_p = right_neighbors[p];
            const std::vector<int> &neighbors_q = right_neighbors[q];
            if (in_neighbor(q, neighbors_p) || in_neighbor(p, neighbors_q))
            {
                right_set.push_back(cv::Point(p, q));
            }
        }
    }

    return left_set.size() + right_set.size();
}

int c_matching_model::get_index(cv::flann::Index &_knn, const cv::Point &_query)
{
    cv::Mat_<float> queries = (cv::Mat_<float>(1, 2) << _query.x, _query.y);
    cv::Mat results, distances;
    _knn.knnSearch(queries, results, distances, 1);
    return *results.data;
}

void c_matching_model::get_knn_results(const cv::Point &_query,
                                       cv::flann::Index &_knn,
                                       std::vector<int> &_indices,
                                       const int _k)
{
    cv::Mat_<float> query = (cv::Mat_<float>(1, 2) << _query.x, _query.y);
    cv::Mat indices, distances;
    _knn.knnSearch(query, indices, distances, _k);
    _indices.clear();
    _indices.reserve(_k - 1);
    //    _distances.reserve(_k);

    /*!< IMPORTANT!! exclude the query point itself from neighbor set */

    for (int i = 1; i < _k; ++i)
    {
        _indices.push_back(indices.at<int>(0, i));
        //        _distances.push_back(distances.at<float>(0, i));
    }
}

bool c_matching_model::in_neighbor_system(t_match_set::const_iterator _cit_a,
                                          t_match_set::const_iterator _cit_b)
{
    const std::vector<std::vector<int> > &left_set =
            m_p_global_data->m_neighbors_left;
    const std::vector<std::vector<int> > &right_set =
            m_p_global_data->m_neighbors_right;
    int p_left = _cit_a->x;
    int q_left = _cit_b->x;
    int p_right = _cit_a->y;
    int q_right = _cit_b->y;

    if (p_left == q_left || p_right == q_right)
    {
        return false;
    }
    if (in_neighbor(q_left, left_set[p_left]))
    {
        return true;
    }
    if (in_neighbor(p_left, left_set[q_left]))
    {
        return true;
    }
    if (in_neighbor(q_right, right_set[p_right]))
    {
        return true;
    }
    if (in_neighbor(p_right, right_set[q_right]))
    {
        return true;
    }

    return false;
}

float c_matching_model::geometric_consistency(t_match_set::const_iterator _cit_a,
                                              t_match_set::const_iterator _cit_b)
{
    const std::vector<cv::Point> &left_points = m_p_global_data->m_features_left;
    const std::vector<cv::Point> &right_points = m_p_global_data->m_features_right;

    const cv::Point &p_left = left_points[_cit_a->x];
    const cv::Point &q_left = left_points[_cit_b->x];
    const cv::Point &p_right = right_points[_cit_a->y];
    const cv::Point &q_right = right_points[_cit_b->y];

    const cv::Point segment_left = p_left - q_left;
    const cv::Point segment_right = p_right - q_right;

    const float length_left = cv::norm(segment_left);
    const float length_right = cv::norm(segment_right);

    assert(length_left && length_right);

    const float delta = std::fabs(length_left - length_right) /
            (length_left + length_right);
    float temp = (segment_left.x * segment_right.x +
                  segment_left.y * segment_right.y) /
                 (length_left * length_right);

    const float alpha = std::acos(std::max(std::min(temp, 1.f), -1.f));

    const float ita = m_p_arguments->m_ita;
    const float sigma_l = m_p_arguments->m_sigma_l;
    const float sigma_a = m_p_arguments->m_sigma_a;

    float result = ita * (std::exp(delta / sigma_l) - 1.f) +
            (1.f - ita) * (std::exp(alpha / sigma_a) - 1.f);

    return result;
}

} // namespace xh
