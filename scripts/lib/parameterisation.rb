require 'yaml'

class Parameterisation
    def initialize(parameterisation)
        @name = parameterisation
    end

    def run_xpp
       vars = load_param
       file = generate_ode(vars)
       puts `xppaut #{file}`
    end

    private
    def load_param
       yaml = {}
       File.open("#{File.dirname(__FILE__)}/../../parameterisations/parameterisations.yml") { |yf| yaml = YAML::load( yf ) }

       baseline = yaml["baseline"]
       param = yaml[@name]
       raise "Unknown parameterisation #{@name}" unless param
       baseline.merge(param)
    end

    def generate_ode(param)
       template = []
       File.open("#{File.dirname(__FILE__)}/../../xppaut/liley_et_al_noisy.ode.template") do |f|
          f.each {|line| template << line}
       end
       template_contents = template.join("\n")
       param.each { |name, val| template_contents= template_contents.gsub("@@#{name.upcase}@@", val.to_s) } 
       file = "#{File.dirname(__FILE__)}/../../tmp/#{@name}.ode"
       File.open(file, 'w') do |f|
           f.puts(template_contents)
       end
       file
    end
end
