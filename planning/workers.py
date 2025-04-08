from pyomo.environ import *

# --------------------------
# Parameters and Sets
# --------------------------
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
shifts = ["day", "night"]
stations = list(range(4))  # Number of parallel workstations
workers = [f"W{i}" for i in range(1, 11)]  # Shared global worker pool

# --------------------------
# Model Initialization
# --------------------------
model = ConcreteModel()

# Binary variable: whether a worker is assigned to a shift at a station
model.works = Var(
    (
        (worker, station, day, shift)
        for worker in workers
        for station in stations
        for day in days
        for shift in shifts
    ),
    within=Binary,
    initialize=0,
)

# Binary variable: whether a worker is needed (for cost/objective visibility)
model.needed = Var(workers, within=Binary, initialize=0)

# --------------------------
# Track the total hours worked by each worker
model.total_hours = Var(workers, within=NonNegativeReals, initialize=0)

# --------------------------
# Track deviation from average worked hours
model.deviation = Var(workers, within=NonNegativeReals)

# --------------------------
# Objective: Maximize number of filled shifts and minimize workload disparity
# --------------------------


# Calculate average total hours worked per worker
average_hours = len(stations) * len(days) * len(shifts) / len(workers)

# Objective function: Maximize filled shifts + Minimize deviation in hours worked
model.obj = Objective(
    expr=(
        sum(
            model.works[worker, station, day, shift]
            for worker in workers
            for station in stations
            for day in days
            for shift in shifts
        )  # Maximizing the number of filled shifts
        - 10
        * sum(
            model.deviation[worker] for worker in workers
        )  # Minimize the deviation from average hours
    ),
    sense=maximize,
)

# --------------------------
# Constraints
# --------------------------
model.constraints = ConstraintList()

# Calculate total shifts worked by each worker
for worker in workers:
    model.constraints.add(
        model.total_hours[worker]
        == sum(
            model.works[worker, station, day, shift]
            for station in stations
            for day in days
            for shift in shifts
        )
    )
# (1) A worker cannot work in more than one station for the same shift
for day in days:
    for shift in shifts:
        for worker in workers:
            model.constraints.add(
                sum(model.works[worker, station, day, shift] for station in stations)
                <= 1
            )

# (2) Day and night shift must be paired for a worker at a given station and day
for station in stations:
    for day in days:
        for worker in workers:
            model.constraints.add(
                model.works[worker, station, day, "day"]
                == model.works[worker, station, day, "night"]
            )

# (3) A worker cannot work two consecutive days
for i in range(len(days)):
    for worker in workers:
        model.constraints.add(
            sum(
                model.works[worker, station, days[i], shift]
                + model.works[
                    worker, station, days[i + 1 if i < len(days) - 1 else 0], shift
                ]
                for station in stations
                for shift in shifts
            )
            <= 2
        )

# (3bis) No more than 1 worker at each station/shift
for station in stations:
    for day in days:
        for shift in shifts:
            model.constraints.add(
                sum(model.works[worker, station, day, shift] for worker in workers) <= 2
            )

# (4) Deviation calculation
for worker in workers:
    model.constraints.add(
        model.deviation[worker] >= model.total_hours[worker] - average_hours
    )
    model.constraints.add(
        model.deviation[worker] >= average_hours - model.total_hours[worker]
    )

# (5) Set 'needed' to 1 if a worker is assigned anywhere
max_possible_shifts = len(stations) * len(days) * len(shifts)
for worker in workers:
    total_assignments = sum(
        model.works[worker, station, day, shift]
        for station in stations
        for day in days
        for shift in shifts
    )
    model.constraints.add(
        model.needed[worker] * max_possible_shifts >= total_assignments
    )

# --------------------------
# Solve
# --------------------------
opt = SolverFactory("cbc")  # Use CBC for MILP problems
opt.options["sec"] = 10
results = opt.solve(model, tee=True)


# --------------------------
# Helper Functions
# --------------------------
def get_workers_needed(needed):
    return [worker for worker in workers if needed[worker].value == 1]


def get_work_table(works):
    schedule = {
        station: {day: {shift: [] for shift in shifts} for day in days}
        for station in stations
    }
    for worker in workers:
        for station in stations:
            for day in days:
                for shift in shifts:
                    if works[worker, station, day, shift].value == 1:
                        schedule[station][day][shift].append(worker)
    return schedule


# --------------------------
# Output
# --------------------------
workers_needed = get_workers_needed(model.needed)
week_table = get_work_table(model.works)

print("\nScheduled Workers Per Shift per Station:")
for station in stations:
    print(f"\nStation {station}")
    for day in days:
        for shift in shifts:
            print(f"{day} - {shift}: {week_table[station][day][shift]}")

# Worker-based Schedule:
print("\n\nWorker-based Schedule:")
for worker in workers:
    print(f"\n{worker}:")
    has_assignment = False
    for station in stations:
        for day in days:
            for shift in shifts:
                if model.works[worker, station, day, shift].value == 1:
                    print(f"  Station {station} - {day} {shift}")
                    has_assignment = True
    if not has_assignment:
        print("  (No shifts assigned)")
