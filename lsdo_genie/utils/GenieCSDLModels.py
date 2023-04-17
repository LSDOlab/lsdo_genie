import csdl
import numpy as np

class Genie2DCSDLModel(csdl.CustomExplicitOperation):
    '''
    Computational System Design Language (CSDL) interface for using `lsdo_genie` in gradient-based
    optimization for 2D geometric shapes

    Parameters
    ----------
    num_pts : int
        Number of points in the input
    x_name : str
        CSDL variable name of the x-coordinate of the input points
    y_name : str
        CSDL variable name of the y-coordinate of the input points
    out_name : str
        CSDL variable name of the output phi
    genie_object : lsdo_genie.Genie2D
        The genie object to be used in the model to compute phi
    '''

    def initialize(self):
        '''
        initialize
        '''
        self.parameters.declare("num_pts",  types=int)
        self.parameters.declare("x_name",   types=str)
        self.parameters.declare("y_name",   types=str)
        self.parameters.declare("out_name", types=str)
        self.parameters.declare("genie_object")

    def define(self):
        '''
        define
        '''
        num_pts = self.parameters["num_pts"]
        x_name = self.parameters["x_name"]
        y_name = self.parameters["y_name"]
        out_name = self.parameters["out_name"]

        self.add_input(x_name,shape=(num_pts,))
        self.add_input(y_name,shape=(num_pts,))
        self.add_output(out_name,shape=(num_pts,))

        self.declare_derivatives(of=out_name,wrt=x_name,rows=np.arange(num_pts),cols=np.arange(num_pts))
        self.declare_derivatives(of=out_name,wrt=y_name,rows=np.arange(num_pts),cols=np.arange(num_pts))

    def compute(self, inputs, outputs):
        '''
        compute
        '''
        x_name = self.parameters["x_name"]
        y_name = self.parameters["y_name"]
        out_name = self.parameters["out_name"]
        genie_obj = self.parameters["genie_object"]

        pts = np.stack((inputs[x_name],inputs[y_name]),axis=1)
        outputs[out_name] = genie_obj.compute_phi(pts)

    def compute_derivatives(self, inputs, derivatives):
        '''
        compute_derivatives
        '''
        x_name = self.parameters["x_name"]
        y_name = self.parameters["y_name"]
        out_name = self.parameters["out_name"]
        genie_obj = self.parameters["genie_object"]

        pts = np.stack((inputs[x_name],inputs[y_name]),axis=1)
        dx,dy = genie_obj.gradient_phi(pts)

        derivatives[out_name,x_name] = dx
        derivatives[out_name,y_name] = dy


class Genie3DCSDLModel(csdl.CustomExplicitOperation):
    '''
    Computational System Design Language (CSDL) interface for using `lsdo_genie` in gradient-based
    optimization for 3D geometric shapes

    Parameters
    ----------
    num_pts : int
        Number of points in the input
    x_name : str
        CSDL variable name of the x-coordinate of the input points
    y_name : str
        CSDL variable name of the y-coordinate of the input points
    z_name : str
        CSDL variable name of the z-coordinate of the input points
    out_name : str
        CSDL variable name of the output phi
    genie_object : lsdo_genie.Genie2D
        The genie object to be used in the model to compute phi
    '''

    def initialize(self):
        '''
        initialize
        '''
        self.parameters.declare("num_pts",  types=int)
        self.parameters.declare("x_name",   types=str)
        self.parameters.declare("y_name",   types=str)
        self.parameters.declare("z_name",   types=str)
        self.parameters.declare("out_name", types=str)
        self.parameters.declare("genie_object")

    def define(self):
        '''
        define
        '''
        num_pts = self.parameters["num_pts"]
        x_name = self.parameters["x_name"]
        y_name = self.parameters["y_name"]
        z_name = self.parameters["z_name"]
        out_name = self.parameters["out_name"]

        self.add_input(x_name,shape=(num_pts,))
        self.add_input(y_name,shape=(num_pts,))
        self.add_input(z_name,shape=(num_pts,))
        self.add_output(out_name,shape=(num_pts,))

        self.declare_derivatives(of=out_name,wrt=x_name,rows=np.arange(num_pts),cols=np.arange(num_pts))
        self.declare_derivatives(of=out_name,wrt=y_name,rows=np.arange(num_pts),cols=np.arange(num_pts))
        self.declare_derivatives(of=out_name,wrt=z_name,rows=np.arange(num_pts),cols=np.arange(num_pts))

    def compute(self, inputs, outputs):
        '''
        compute
        '''
        x_name = self.parameters["x_name"]
        y_name = self.parameters["y_name"]
        z_name = self.parameters["z_name"]
        out_name = self.parameters["out_name"]
        genie_obj = self.parameters["genie_object"]

        pts = np.stack((inputs[x_name],inputs[y_name],inputs[z_name]),axis=1)
        outputs[out_name] = genie_obj.compute_phi(pts)

    def compute_derivatives(self, inputs, derivatives):
        '''
        compute_derivatives
        '''
        x_name = self.parameters["x_name"]
        y_name = self.parameters["y_name"]
        z_name = self.parameters["z_name"]
        out_name = self.parameters["out_name"]
        genie_obj = self.parameters["genie_object"]

        pts = np.stack((inputs[x_name],inputs[y_name],inputs[z_name]),axis=1)
        dx,dy,dz = genie_obj.gradient_phi(pts)

        derivatives[out_name,x_name] = dx
        derivatives[out_name,y_name] = dy
        derivatives[out_name,z_name] = dz