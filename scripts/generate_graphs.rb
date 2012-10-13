#!/bin/ruby

type = ARGV[0]
prefix = ARGV[1]

Dir["parameterisations/#{prefix}**"].each do |file|
    puts file
    puts `python python/display_#{type}.py #{file} save`
end
