require 'yaml'
require 'pp'

class ParamSetAnalyser
  def initialize(root_dir)
    @root_dir = root_dir

    @groups = {}
  end

  def run
    Dir["#{@root_dir}/**/*.yml"].each do |yml_file|
      index_details(yml_file) unless File.directory?(yml_file)
    end

    puts "e_ii, e_ie, g_e, g_i, mus_i, mus_i"
    pp @groups.sort { |a, b| b.last.length <=> a.last.length }
  end

  def index_details(yml_file)
    params = YAML.load_file(yml_file)

    params.each do |key, value|
      group = [value['e_ii'], value['e_ie'], value['g_e'], value['g_i'], value['mus_i'], value['mus_i']]
      @groups[group] ||= []
      @groups[group] << yml_file
    end
  end
end


ParamSetAnalyser.new(ARGV[0]).run