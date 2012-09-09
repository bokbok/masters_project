Dir["parameterisations/**/*.ode-burst.yml"].each do |f|
   puts `python python/display_intracellular.py #{f} save`
end
