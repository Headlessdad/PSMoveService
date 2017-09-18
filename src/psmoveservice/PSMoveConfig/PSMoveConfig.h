#ifndef PSMOVE_CONFIG_H
#define PSMOVE_CONFIG_H

//-- includes -----
#include <string>
#include <array>
#include <boost/property_tree/ptree.hpp>

//-- constants -----
extern const struct CommonHSVColorRange *k_default_color_presets;

//-- definitions -----
class PSMoveConfig {
public:
    PSMoveConfig(const std::string &fnamebase = std::string("PSMoveConfig"));
    void save();
    bool load();
    
    std::string ConfigFileBase;

    virtual const boost::property_tree::ptree config2ptree() = 0;  // Implement by each device class' own Config
    virtual void ptree2config(const boost::property_tree::ptree &pt) = 0;  // Implement by each device class' own Config
    
    static void writeDistortionCoefficients(
        boost::property_tree::ptree &pt,
        const char *coefficients_name,
        const struct CommonDistortionCoefficients *coefficients);
    static void readDistortionCoefficients(
        const boost::property_tree::ptree &pt,
        const char *coefficients_name,
        struct CommonDistortionCoefficients *outCoefficients,
        const struct CommonDistortionCoefficients *defaultCoefficients);

    static void writeColorPreset(
        boost::property_tree::ptree &pt,
        const char *profile_name,
        const char *color_name,
        const struct CommonHSVColorRange *colorPreset);
    static void readColorPreset(
        const boost::property_tree::ptree &pt,
        const char *profile_name,
        const char *color_name,
        struct CommonHSVColorRange *outColorPreset,
        const struct CommonHSVColorRange *defaultPreset);

	static void writeColorPropertyPresetTable(
		const struct CommonHSVColorRangeTable *table,
		boost::property_tree::ptree &pt);
	static void readColorPropertyPresetTable(
		const boost::property_tree::ptree &pt,
		struct CommonHSVColorRangeTable *table);

    template <typename T, size_t s>
    std::array<T, s> readArray(boost::property_tree::ptree const& pt,  boost::property_tree::ptree::key_type const& key)
    {
        std::array<T, s> r;
        size_t write_count= 0;
        for (auto& item : pt.get_child(key))
        {
            if (write_count < s)
            {
                r[write_count] = item.second.get_value<T>();
                write_count++;
            }
            else
            {
                break;
            }
        }
        return r;
    }

    template <typename T, size_t s>
    void writeArray(
        boost::property_tree::ptree& pt, 
        boost::property_tree::ptree::key_type const& key,
        const std::array<T, s> &values)
    {
        boost::property_tree::ptree ptArray;
        boost::property_tree::ptree ptElement;

        for (int i = 0; i < s; ++i)
        {
            ptElement.put_value(values[i]);
            ptArray.push_back(std::make_pair("", ptElement));
        }
        pt.put_child(key, ptArray);
    }

	static void writeTrackingColor(boost::property_tree::ptree &pt, int tracking_color_id);
	static int readTrackingColor(const boost::property_tree::ptree &pt);

private:
    const std::string getConfigPath();
};
/*
Note that PSMoveConfig is an abstract class because it has 2 pure virtual functions.
Child classes must add public member variables that store the config data,
as well as implement config2ptree and ptree2config that use pt.put() and
pt.get(), respectively, to convert between member variables and the
property tree. See tests/test_config.cpp for an example.
*/
#endif // PSMOVE_CONFIG_H