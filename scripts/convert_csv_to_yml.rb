require "csv"
require "yaml"

class Convert
  FIELDS = %w{h_e_rest h_i_rest tor_e tor_i h_ee_eq h_ei_eq h_ie_eq h_ii_eq T_ee T_ei T_ie T_ii gamma_ee gamma_ei gamma_ie gamma_ii N_beta_ee N_beta_ei N_beta_ie N_beta_ii N_alpha_ee N_alpha_ei A v s_e_max s_i_max mu_e mu_i sigma_e sigma_i p_ee p_ei}
  def initialize

  end

  def convert
    params= {}
    num = 0
    CSV.foreach(File.dirname(__FILE__) + "/../parameterisations/biphasic.csv") do |row|
      set = {}

      FIELDS.each_with_index do |field, i|
        unless field == "A"
          set[field] = row[i].to_f
        else
          set["A_ee"] = row[i].to_f
          set["A_ei"] = row[i].to_f
        end
      end

      set['r_abs'] = 0
      set['phi_ie'] = 0
      set['phi_ii'] = 0
      set['p_ii'] = 0
      set['p_ie'] = 0

      params["biphasic#{num}"] = set
      num += 1
    end

    File.open(File.dirname(__FILE__) + "/../parameterisations/biphasic.yml", "w") do |file|
      YAML.dump(params, file)
    end
  end
end

Convert.new.convert