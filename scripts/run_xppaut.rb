#!/usr/bin/ruby
require "optparse"

Dir["#{File.dirname(__FILE__)}/lib/**/*.rb"].each do |file|
    require file
end

options = {}
OptionParser.new do |opts|
  opts.banner = "Usage: example.rb [options]"

  opts.on("-m", "--model NAME", "Select model") do |m|
    options[:model] = m 
  end
  opts.on("-p", "--parameterisation NAME", "Select parameterisation") do |p|
    options[:parameterisation] = p
  end
end.parse!

param = Parameterisation.new(options)
param.run_xpp
