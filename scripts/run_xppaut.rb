#!/usr/bin/ruby
require "optparse"

Dir["#{File.dirname(__FILE__)}/lib/**/*.rb"].each do |file|
    require file
end

options = {}
OptionParser.new do |opts|
  opts.banner = "Usage: example.rb [options]"

  opts.on("-p", "--parameterisation NAME", "Select parameterisation") do |p|
    options[:parameterisation] = p
  end
end.parse!

param = Parameterisation.new(options[:parameterisation])
param.run_xpp
