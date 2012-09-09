Dir["parameterisations/**/*.ode-phi-burst.yml"].each do |f|
   puts `python python/display_phi.py #{f} save`
end
