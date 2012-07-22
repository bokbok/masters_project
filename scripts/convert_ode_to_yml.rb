require "yaml"

class Convert
  TRANSLATIONS = {
      'her' => 'h_e_rest',
      'hir' => 'h_i_rest',
      'hir' => 'h_i_rest',
      'heqee' => 'h_ee_eq',
      'heqei' => 'h_ei_eq',
      'heqie' => 'h_ie_eq',
      'heqii' => 'h_ii_eq',
      'naee' => 'N_alpha_ee',
      'naei' => 'N_alpha_ei',
      'nbee' => 'N_beta_ee',
      'nbei' => 'N_beta_ei',
      'nbie' => 'N_beta_ie',
      'nbii' => 'N_beta_ii',
      'lambdaee' => 'A_ee',
      'lambdaei' => 'A_ei',
      'vbar' => 'v',
      'aee' => "T_ee",
      'aei' => "T_ei",
      'aie' => "T_ie",
      'aii' => "T_ii",
      'gee' => 'gamma_ee',
      'gei' => 'gamma_ei',
      'gie' => 'gamma_ie',
      'gii' => 'gamma_ii',
      'taue' => 'tor_e',
      'taui' => 'tor_i',
      'te' => 'mu_e',
      'ti' => 'mu_i',
      'se' => 'sigma_e',
      'si' => 'sigma_i',
      'emax' => 's_e_max',
      'imax' => 's_i_max',
      'pii' => 'p_ii',
      'pie' => 'p_ie',
      'pei' => 'p_ei',
      'pee' => 'p_ee'
  }

  def initialize(dir, out)
    @dir = dir
    @out = out
  end

  def run
    params = {}
    Dir["#{@dir}/**/*.ode"].each do |file|
      params[File.basename(file)] = extract_params(file)
    end

    File.open("#{@out}/#{File.basename(File.dirname(@dir))}.yml", "w+") do |f|
      YAML.dump(params, f)
    end
  end

  def extract_params(file)
    params = {}
    File.open(file) do |f|
      f.readlines.each do |line|
        if line =~ /^\s*param\s*(.*)=(\d+\.{0,1}\d*).*$/
          params[TRANSLATIONS[$1] || $1] = $2.to_f
        end
      end
    end
    params
  end
end

Convert.new(ARGV[0], ARGV[1]).run