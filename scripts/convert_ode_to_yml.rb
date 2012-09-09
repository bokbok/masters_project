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
      'pee' => 'p_ee',
      'rabs' => 'r_abs'
  }

  def initialize(dir, out, field)
    @dir = dir
    @out = out
    @field = field
  end

  def run
    params = { }
    Dir["#{@dir}/**/*.ode"].each do |file|
      params[File.basename(file)] = extract_params(file)
    end

    File.open("#{@out}/#{File.basename(@dir)}.yml", "w+") do |f|
      YAML.dump(params, f)
    end
  end

  def extract_params(file)
    params = {'r_abs' => 0,
              'tor_slow' => 1,
              'g' => 0,
              'burst_i' => 0,
              'burst_e' => 0,
              'phi_ii' => 0,
              'phi_ie' => 0}

    File.open(file) do |f|
      f.readlines.each do |line|
        if line =~ /^\s*param\s*(.*)=(-{0,1}\d+\.{0,1}\d*).*$/
          params[TRANSLATIONS[$1] || $1] = $2.to_f
        end
      end
    end

    adjust_params(params, file)
    params
  end

  private
  def adjust_params(params, file)
    hb_files = Dir["#{file.gsub(".ode", "")}/**/hb-*"]

    if hb_files.length == 2
      find_midpoint(hb_files, params)
    else
      move_beyond_bif(hb_files.first, file, params)
    end
  end

  def move_beyond_bif(file, process_file, params)
    if file
      bif_params = read_file(file)
      dir = bif_params[@field] - params[@field]
      if dir > 0
        dir = 1
      else
        dir = -1
      end

      params[@field] = bif_params[@field] + dir * 0.1

      bif_params.each do |k, v|
        params[k] = v unless k == @field
      end

    else
      puts "No hopf point for #{process_file}"
    end
  end

  def find_midpoint(hb_files, params)
    bif_params = hb_files.map{ |file| read_file(file) }
    mid = bif_params.first[@field] + (bif_params.last[@field] - bif_params.first[@field]) / 2

    params[@field] = mid

    bif_params.last.each do |k, v|
      params[k] = v unless k == @field
    end
  end

  def read_file(file)
    res = {}
    File.open(file, 'r') do |f|
      lines = f.readlines
      lines.each do |line|
        kv = line.strip.split("=")
        res[kv.first] = kv.last.to_f
      end
    end

    res
  end
end

Convert.new(ARGV[0], ARGV[1], ARGV[2]).run
