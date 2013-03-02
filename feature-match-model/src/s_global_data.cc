#include "s_global_data.hh"

#include <fstream>

namespace xh
{

s_global_data *s_global_data::m_p_instance = NULL;

s_global_data::s_global_data(void)
{
}

bool s_global_data::load(const std::string &_file_name)
{
    return true;
}

bool s_global_data::save(const std::string &_file_name)
{
    std::ofstream file(_file_name.c_str());
    file.clear();
    // to do:

    file.close();
    return true;
}

} // namespace xh
