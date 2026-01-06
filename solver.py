import time
import numpy as np
import gurobipy


def solver(theta, MODEL, args, legalList):

    import gurobipy
    import time
    s = time.time()

    variables = []
    for i in range(args.solDim * args.card):
        variables.append(MODEL.addVar(vtype=gurobipy.GRB.BINARY, name='x' + 'i'))

    MODEL.update()

    MODEL.setObjective(np.array(variables).dot(theta), sense=gurobipy.GRB.MAXIMIZE)

    cnt = 0
    for i in range(args.solDim):
        if len(legalList) > 0:
            for j in legalList[i]:
                tmp = 0
                for line in range(args.card):
                    tmp = tmp - variables[line * args.solDim + i] - variables[line * args.solDim + j]

                MODEL.addConstr(tmp >= -1, name=str(cnt))
                cnt += 1
    MODEL.addConstr(sum(variables) == args.card, name=str(cnt))
    f = time.time()

    MODEL.optimize()
    return np.array(MODEL.x)


def solver_quad(Q, MODEL, args, legalList):

    import gurobipy
    import time
    s = time.time()

    variables = []
    for i in range(args.solDim * args.card):
        variables.append(MODEL.addVar(vtype=gurobipy.GRB.BINARY, name='x' + 'i'))

    MODEL.update()

    MODEL.setObjective(np.array(variables).dot(Q).dot(np.array(variables)), sense=gurobipy.GRB.MAXIMIZE)

    cnt = 0
    for i in range(args.solDim):
        if len(legalList) > 0:
            for j in legalList[i]:
                tmp = 0
                for line in range(args.card):
                    tmp = tmp - variables[line * args.solDim + i] - variables[line * args.solDim + j]

                MODEL.addConstr(tmp >= -1, name=str(cnt))
                cnt += 1
    MODEL.addConstr(sum(variables) == args.card, name=str(cnt))
    f = time.time()

    MODEL.optimize()
    return np.array(MODEL.x)


def solver_mixed(Q, a, MODEL, args, legalList):
    s = time.time()

    variables = []
    for i in range(args.solDim * args.card):
        variables.append(MODEL.addVar(vtype=gurobipy.GRB.BINARY, name='x' + 'i'))

    MODEL.update()

    MODEL.setObjective(np.array(variables).dot(Q).dot(np.array(variables)) + (np.array(variables).dot(a)) * (np.array(variables).dot(a)), sense=gurobipy.GRB.MAXIMIZE)

    cnt = 0
    for i in range(args.solDim):
        if len(legalList) > 0:
            for j in legalList[i]:
                tmp = 0
                for line in range(args.card):
                    tmp = tmp - variables[line * args.solDim + i] - variables[line * args.solDim + j]

                MODEL.addConstr(tmp >= -1, name=str(cnt))
                cnt += 1
    MODEL.addConstr(sum(variables) == args.card, name=str(cnt))
    f = time.time()
    MODEL.optimize()
