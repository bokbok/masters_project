require 'yaml'
require 'fileutils'

class ConvertYamlToOde
  TRANSLATIONS = {
      'h_e_rest' => 'her',
      'h_i_rest' => 'hir',
      'h_i_rest' => 'hir',
      'h_ee_eq' => 'heqee',
      'h_ei_eq' => 'heqei',
      'h_ie_eq' => 'heqie',
      'h_ii_eq' => 'heqii',
      'N_alpha_ee' => 'naee',
      'N_alpha_ei' => 'naei',
      'N_beta_ee' => 'nbee',
      'N_beta_ei' => 'nbei',
      'N_beta_ie' => 'nbie',
      'N_beta_ii' => 'nbii',
      'A_ee' => 'lambdaee',
      'A_ei' => 'lambdaei',
      'v' => 'vbar',
      "T_ee" => 'aee',
      "T_ei" => 'aei',
      "T_ie" => 'aie',
      "T_ii" => 'aii',
      'gamma_ee' => 'gee',
      'gamma_ei' => 'gei',
      'gamma_ie' => 'gie',
      'gamma_ii' => 'gii',
      'tor_e' => 'taue',
      'tor_i' => 'taui',
      'mu_e' => 'te',
      'mu_i' => 'ti',
      'sigma_e' => 'se',
      'sigma_i' => 'si',
      's_e_max' => 'emax',
      's_i_max' => 'imax',
      'p_ii' => 'pii',
      'p_ie' => 'pie',
      'p_ei' => 'pei',
      'p_ee' => 'pee',
      'r_abs' => 'rabs',
      'e_ee' => 'epsilonee',
      'e_ei' => 'epsilonei',
      'e_ie' => 'epsilonie',
      'e_ii' => 'epsilonii'
  }

  def initialize(set_dir, out_dir)
    @set_dir = set_dir
    @out_dir = out_dir + File.basename(@set_dir)
  end

  def convert_siru3_dir
    puts "HERE #{@set_dir}"
    Dir["#{@set_dir}/**/*.yml"].each do |yml_file|
      puts yml_file
      convert_siru3(yml_file)
    end
    puts "DONE"
  end

  def convert_siru3(yml_file)
    vals = YAML.load_file(yml_file)["params"]

    template = File.read("#{File.dirname(__FILE__)}/siru3_template.ode")

    vals["e_ee"] ||= 1e-6
    vals["e_ie"] ||= 1e-6
    vals["e_ii"] ||= 1e-6
    vals["e_ei"] ||= 1e-6

    vals.each do |name, val|
      name_converted = TRANSLATIONS[name] || name
      template = template.gsub(/param\s*#{name_converted}\s*=.*$/, "param #{name_converted} = #{val}")
    end

    template = template.gsub(/he\(0\)\s*=.*$/, "he(0) = #{vals['h_e_rest']}")
    template = template.gsub(/hi\(0\)\s*=.*$/, "hi(0) = #{vals['h_i_rest']}")

    FileUtils.mkdir_p(@out_dir)

    File.open(@out_dir + "/" + File.basename(yml_file, '.yml') + ".ode", 'w+') do |file|
      file.puts(template)
    end
  end
end

ConvertYamlToOde.new(ARGV[0], ARGV[1]).convert_siru3_dir