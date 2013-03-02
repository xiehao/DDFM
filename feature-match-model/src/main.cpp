#include <iostream>

#include <c_matching_model.hh>
#include <opencv2/flann/flann.hpp>

#include "s_arguments.hh"
#include "c_matching_model.hh"

using namespace xh;
using namespace std;

//#define SSSSS

typedef std::vector<cv::Point> t_feature_set;
bool build_knn(const t_feature_set &_points, cv::Mat_<float> &_features,
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

#if 1 // for debug
    std::cout << "Points for building knn:" << std::endl;
    std::cout << _features << std::endl;
#endif

    _knn.build(_features, cv::flann::KDTreeIndexParams());

    return true;
}

void test_flann(void)
{
    /*!< randomly generate n points as the whole features */
    const size_t n = 3;

    std::vector<cv::Point> point_set(n);
#ifdef SSSSS
    cv::Mat_<float> features = cv::Mat_<float>::zeros(n, 2);
#endif // SSSSS
#if 0
    point_set[0] = cv::Point(303, 137);
    point_set[1] = cv::Point(353, 164);
    point_set[2] = cv::Point(388, 196);
    features = (cv::Mat_<float>(3, 2) << 303, 137, 353, 164, 388, 196);
#endif
    std::vector<cv::Point>::iterator it = point_set.begin();
    std::vector<cv::Point>::iterator it_end = point_set.end();
    srand(static_cast<unsigned>(time(0)));
    cout << "Point set:" << endl;
    for (int i = 0; it != it_end; ++it, ++i)
    {
#if 1
        *it = cv::Point(rand() % 10, rand() % 10);
#endif
        cout << *it << '\t';
#ifdef SSSSS
        float *p = features.ptr<float>(i);
        p[0] = it->x;
        p[1] = it->y;
#endif // SSSSS
    }
    cout << endl;

    /*!< construct kd tree for further knn search */
    cv::flann::Index knn;
#ifdef SSSSS
    knn.build(features, cv::flann::KDTreeIndexParams());
#else
    cv::Mat_<float> features;
    build_knn(point_set, features, knn);
#endif // SSSSS

    /*!< specify a particular point to search from features */
    cv::Point query_point(point_set[rand() % n]);
    cv::Mat_<float> query = (cv::Mat_<float>(1, 2) << query_point.x, query_point.y);

    /*!< do the searching work */
    cv::Mat indices, distances;
    int k = n;
    knn.knnSearch(query, indices, distances, k);

    /*!< print out searching results */
    cout << "Query point is: " << query << endl;
    cout << "Results are: " << indices << endl;
    cout << "Distances are: " << distances << endl;

    std::vector<int> _indices;
    std::vector<float> _distances;
    _indices.reserve(k);
    _distances.reserve(k);
    for (int i = 1; i < k; ++i)
    {
        _indices.push_back(indices.at<int>(0, i));
        _distances.push_back(distances.at<float>(0, i));
        cout << _indices.back() << '\t' << _distances.back() << endl;
    }
}

void test_map(void)
{
    std::map<int, float> test;
    test[0] += 3.2;
    test[10000] -= 8.7;
    test[10000] += 3.2;
    std::map<int, float>::const_iterator cit = test.begin();
    std::map<int, float>::const_iterator cit_end = test.end();
    for (; cit != cit_end; ++cit)
    {
        std::cout << cit->first << '\t' << cit->second << std::endl;
    }
}

int main(void)
{
//    test_map();
#if 0
    test_flann();
#else
    xh::s_arguments::generate_sample_configuration();
    c_matching_model::demo();
#endif
    cout << "Hello World!" << endl;
    return 0;
}

