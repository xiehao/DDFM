#include "s_arguments.hh"

#include <fstream>
#include <sstream>
#include <vector>

namespace xh
{

s_arguments *s_arguments::m_p_instance = NULL;
const std::string workspace = "../../data/";

s_arguments::s_arguments()
    : m_is_ready(false), m_window_size(31), m_max_disparity(60)
{
}

bool s_arguments::load(const std::string &_file_name)
{
    std::ifstream file(_file_name.c_str());
    if (!file)
    {
        std::cerr << "Loading configuration file failed!!" << std::endl;
        return false;
    }

    std::string line, key, dump;
    for (; std::getline(file, line);)
    {
        std::istringstream stream(line);
        stream >> key >> dump;
        if ('#' == key.c_str()[0])
        {
            std::cout << line << std::endl;
        }
        else if (key == "[left-image]")
        {
            std::string value;
            stream >> value;
            m_left = cv::imread(value.c_str(), CV_LOAD_IMAGE_COLOR);
        }
        else if (key == "[right-image]")
        {
            std::string value;
            stream >> value;
            m_right = cv::imread(value.c_str(), CV_LOAD_IMAGE_COLOR);
        }
        else if (key == "[left-mask]")
        {
            std::string value;
            stream >> value;
            m_left_mask = cv::imread(value.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        }
        else if (key == "[right-mask]")
        {
            std::string value;
            stream >> value;
            m_right_mask = cv::imread(value.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
        }
        else if (key == "[max-disparity]")
        {
            int value;
            stream >> value;
            m_max_disparity = value;
        }
        else if (key == "[window-size]")
        {
            int value;
            stream >> value;
            m_window_size = value;
        }
        else if (key == "[lambda-appearance]")
        {
            float value;
            stream >> value;
            m_lambda_appearance = value;
        }
        else if (key == "[lambda-occlusion]")
        {
            float value;
            stream >> value;
            m_lambda_occlusion = value;
        }
        else if (key == "[lambda-geometry]")
        {
            float value;
            stream >> value;
            m_lambda_geometry = value;
        }
        else if (key == "[lambda-coherence]")
        {
            float value;
            stream >> value;
            m_lambda_coherence = value;
        }
        else if (key == "[ita]")
        {
            float value;
            stream >> value;
            m_ita = value;
        }
        else if (key == "[sigma-l]")
        {
            float value;
            stream >> value;
            m_sigma_l = value;
        }
        else if (key == "[sigma-a]")
        {
            float value;
            stream >> value;
            m_sigma_a = value;
        }
    }

    if (!(m_left.data && m_right.data && m_left_mask.data && m_right_mask.data))
    {
        std::cerr << "Loading images failed!!" << std::endl;
        return false;
    }

    assert(m_right_mask.size() == m_right.size() &&
           m_left_mask.size() == m_left.size() &&
           m_left.size() == m_right.size());

    m_image_size = m_left.size();

    m_is_ready = true;

    return true;
}

bool s_arguments::save(const std::string &_file_name)
{
    std::ofstream file(_file_name.c_str());
    file.clear();

    file << "# Configuration of graph matching for stereo images" << std::endl;

    file << "[left-image]\t\t=\t" << workspace + "L.png" << std::endl;
    file << "[right-image]\t\t=\t" << workspace + "R.png" << std::endl;
    file << "[left-mask]\t\t=\t" << workspace + "L-mask.png" << std::endl;
    file << "[right-mask]\t\t=\t" << workspace + "R-mask.png" << std::endl;
    file << "[max-disparity]\t\t=\t10" << std::endl;
    file << "[window-size]\t\t=\t31" << std::endl;
    file << "[lambda-appearance]\t\t=\t1" << std::endl;
    file << "[lambda-occlusion]\t\t=\t1" << std::endl;
    file << "[lambda-geometry]\t\t=\t1" << std::endl;
    file << "[lambda-coherence]\t\t=\t1" << std::endl;
    file << "[ita]\t\t=\t.5" << std::endl;
    file << "[sigma-l]\t\t=\t1" << std::endl;
    file << "[sigma-a]\t\t=\t1" << std::endl;

    file << "# End of file..." << std::endl;

    file.close();
    return true;
}

} // namespace xh
