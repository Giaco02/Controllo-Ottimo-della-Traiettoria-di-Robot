import casadi as cs
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from example_robot_data.robots_loader import getModelPath, readParamsFromSrdf
from adam.casadi.computations import KinDynComputations
import time
from pinocchio.robot_wrapper import RobotWrapper

class OptimalControlSolver:
    def __init__(self,q_des,tau_method,dt=0.02,N=30, q0=np.zeros(6), dq0=np.zeros(6), joints_name_list=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']):
        self.URDF_path = self.loadUR_urdf()
        self.joints_name_list = joints_name_list
        self.kinDyn = KinDynComputations(self.URDF_path, self.joints_name_list)
        self.H_b = np.eye(4)
        self.v_b = np.zeros(6)
        
        # Limiti dei giunti
        self.effort_limits = self.get_joint_effort_limits()
        self.joint_limits_l = self.get_joint_limits_l()
        self.joint_limits_u = self.get_joint_limits_u()
        
        # Altri parametri
        self.dt = dt
        self.N = N
        self.nq = len(self.joints_name_list)
        self.q0 = q0
        self.dq0 = dq0
        self.q_des = q_des
        self.tau_method = tau_method
    
    def loadUR_urdf(self, robot=5, limited=False, gripper=False):
        assert (not (gripper and (robot == 10 or limited)))
        URDF_FILENAME = "ur%i%s_%s.urdf" % (robot, "_joint_limited" if limited else '', 'gripper' if gripper else 'robot')
        URDF_SUBPATH = "/ur_description/urdf/" + URDF_FILENAME
        modelPath = getModelPath(URDF_SUBPATH)
        try:        
            path = '/opt/openrobots/share/'
            model = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [path])
            if robot == 5 or robot == 3 and gripper:
                SRDF_FILENAME = "ur%i%s.srdf" % (robot, '_gripper' if gripper else '')
                SRDF_SUBPATH = "/ur_description/srdf/" + SRDF_FILENAME
                readParamsFromSrdf(model, modelPath + SRDF_SUBPATH, False, False, None)
            return modelPath + URDF_SUBPATH
        except:
            return modelPath + URDF_SUBPATH

    def get_joint_effort_limits(self):
        tree = ET.parse(self.URDF_path)
        root = tree.getroot()
        effort_limits = []
        for joint in root.findall('joint'):
            limit = joint.find('limit')
            if limit is not None:
                effort = limit.get('effort', None) 
                if effort is not None:
                    effort_limits.append(float(effort))
        return effort_limits
    
    def get_joint_limits_l(self):
        tree = ET.parse(self.URDF_path)
        root = tree.getroot()
        joint_limits_l = []
        for joint in root.findall('joint'):
            limit = joint.find('limit')
            if limit is not None:
                lower = limit.get('lower', None)
                joint_limits_l.append(float(lower))
        return joint_limits_l

    def get_joint_limits_u(self):
        tree = ET.parse(self.URDF_path)
        root = tree.getroot()
        joint_limits_u = []
        for joint in root.findall('joint'):
            limit = joint.find('limit')
            if limit is not None:
                upper = limit.get('upper', None)
                joint_limits_u.append(float(upper))
        return joint_limits_u
    
    def setup_optimization_problem(self):
        # Variabili di ottimizzazione
        q = cs.SX.sym('q', self.nq)
        dq = cs.SX.sym('dq', self.nq)
        ddq = cs.SX.sym('ddq', self.nq)
        states = cs.vertcat(q, dq)
        control = ddq
        rhs = cs.vertcat(dq, ddq)
        self.f = cs.Function('f', [states, control], [rhs])
        
        U = cs.SX.sym('U', self.nq, self.N)
        X = cs.SX.sym('X', self.nq * 2, self.N + 1)
        P = cs.SX.sym('P', 18)
        
        # Trovo soluzione simbolica
        X[:, 0] = P[0:12]
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            f_value = self.f(st, con)
            st_next = st + self.dt * f_value
            X[:, k + 1] = st_next
        
        self.ff = cs.Function('ff', [U, P], [X])
        
        # Creo la funzione di costo
        w_v = 1e-4  # peso sulla velocità
        w_a = 1e-6  # peso sull'accelerazione 
        w_final = 1e2  # peso del costo finale
        
        cost = 0
        for k in range(self.N): 
            cost += cs.mtimes((X[0:6, k] - P[12:18]).T, (X[0:6, k] - P[12:18])) 
            cost += w_v * cs.mtimes(X[6:12, k].T, X[6:12, k])
            cost += w_a * cs.mtimes(U[:, k].T, U[:, k])
        
        # Aggiungo il costo sullo stato finale
        cost += w_final * cs.mtimes((X[0:6, self.N] - P[12:18]).T, (X[0:6, self.N] - P[12:18])) 
        cost += w_final * cs.mtimes(X[6:12, self.N].T, X[6:12, self.N])
        
        g = []  # contiene i vincoli
        # Vincoli su tau con scelta del metodo
        for k in range(self.N):
            if self.tau_method == 'crba':
                tau = (self.kinDyn.rbdalgos.rnea(self.H_b, X[0:6, k].T,
                                                 self.v_b, X[6:12, k].T,
                                                 0 * U[:, k].T, 
                                                 self.kinDyn.g)).array[6:]
                M = self.kinDyn.rbdalgos.crba(self.H_b, X[0:6, k].T)[0].array
                tau += cs.mtimes(M[6:,6:], U[:, k])
            elif self.tau_method == 'rnea':
                tau = (self.kinDyn.rbdalgos.rnea(self.H_b, X[0:6, k].T,
                                                 self.v_b, X[6:12, k].T,
                                                 U[:, k].T, 
                                                 self.kinDyn.g)).array[6:]

            for i in range(6):
                g.append(tau[i])

            if k == 0:
                self.inv_dyn = cs.Function('inv_dyn', [X[0:6, k], X[6:12, k], U[:, k]], [tau])
        
        # Vincoli sulla posizione 
        for k in range(0, self.N + 1):
            q_aux = X[0:6, k]
            for i in range(6):
                g.append(q_aux[i])
        
        # Creo problema di ottimizzazione
        OPT_variables = cs.reshape(U, -1, 1)
        self.nlp = {'x': OPT_variables,
                    'f': cost, 
                    'g': cs.vertcat(*g),
                    'p': P}
        opts = {
            "ipopt.print_level": 2,
            "ipopt.tol": 1e-6,
            "ipopt.constr_viol_tol": 1e-14,
            "ipopt.compl_inf_tol": 1e-14
        }
        self.solver = cs.nlpsol('solver', 'ipopt', self.nlp, opts)
    
    def solve(self):
        # Configuro i limiti dei vincoli
        args = {
            'lbg': np.zeros((12 * self.N + 6, 1)),
            'ubg': np.zeros((12 * self.N + 6, 1))
        }
        # Limiti per tau
        for i in range(6):
            args['lbg'][i:6*self.N:6] = -self.effort_limits[i]
            args['ubg'][i:6*self.N:6] = self.effort_limits[i]
        
        # Limiti per la posizione
        for i in range(6):
            args['lbg'][6*self.N+i:12*self.N+6:6] = self.joint_limits_l[i]
            args['ubg'][6*self.N+i:12*self.N+6:6] = self.joint_limits_u[i]
        
        # Risoluzione del problema di ottimizzazione
        p = np.concatenate([self.q0, self.dq0, self.q_des]).T
        sol = self.solver(p=p, lbg=args['lbg'], ubg=args['ubg'])
        return sol
    
    def get_solution(self, sol):
        ddq_sol = cs.reshape(sol['x'][:, :], self.nq, self.N)
        x_sol = self.ff(ddq_sol, np.concatenate([self.q0, self.dq0, self.q_des]).T)
        q_sol = x_sol[0:6, :]
        dq_sol = x_sol[6:12, :]
        return q_sol, dq_sol
    
    def plot_results(self, q_sol):
        x = np.arange(0, (self.N + 1) * self.dt, self.dt)
        plt.figure(figsize=(10, 6))

        for i in range(q_sol.shape[0]):
            plt.plot(x, q_sol[i, :].T, label=f'q {i}')

        plt.xlabel('Tempo (s)')
        plt.ylabel('Valori(rad)')
        plt.title('Grafico delle Sei Configurazioni')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def run(self):
        start_time = time.time()
        # Imposta il problema di ottimizzazione
        self.setup_optimization_problem()
        sol = self.solve()
        q_sol, dq_sol = self.get_solution(sol)
        end_time = time.time()
        
        print(f"Tempo di esecuzione: {end_time - start_time} secondi")
        
        for i in range(self.N):
            print("tau_%d =" % i, self.inv_dyn(q_sol[:, i], dq_sol[:, i], sol['x'][self.nq*i:self.nq*(i+1)]))
        
        print("q desiderata:", self.q_des)
        print("q finale:    ", q_sol[:, self.N])
        print("dq finale:   ", dq_sol[:, self.N])
        for i in range(self.N):
            formatted_array = np.array2string(q_sol[:, i].full(), precision=12)
            print(f"q ({i}): {formatted_array}")


        
        self.plot_results(q_sol)

    def time(self):
        start_time = time.time()
        # Imposta il problema di ottimizzazione
        self.setup_optimization_problem()
        sol = self.solve()
        q_sol, dq_sol = self.get_solution(sol)
        end_time = time.time()
        return(end_time - start_time)