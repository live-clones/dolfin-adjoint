from __future__ import print_function
from mshr import *
from dolfin import *

# Define geometry scales
R = 2
c = 5
b = 1 # Blade length
w = 0.1 # Relative width
resolution_back = 20
resolution_front = 40

# Background mesh
circle = Circle(Point(c,c), R)
rectangle = Rectangle(Point(0,0), Point(10,10))
back_geo = rectangle - circle
back_mesh = generate_mesh(back_geo, resolution_back)

# Mesh of Propeller
circle_inner = Circle(Point(c,c), 1.3*R)
stem = Circle(Point(c,c), 0.1*R)
blade_north = Ellipse(Point(c, c+0.5*b), b*w, 0.5*b)
blade_west = Ellipse(Point(c-0.5*b,c), 0.5*b, w*b)
blade_south = Ellipse(Point(c, c-0.5*b), b*w, 0.5*b)
blade_east = Ellipse(Point(c+0.5*b,c), 0.5*b, w*b)
turbine = circle_inner - stem - blade_north - blade_west - blade_south\
          - blade_east
front_mesh = generate_mesh(turbine, resolution_front)

# Save to file
File("propeller_background.xml.gz") << back_mesh
File("propeller_front.xml.gz") << front_mesh

# Visualization
plot(front_mesh)
plot(back_mesh)
interactive()
