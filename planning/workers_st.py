import streamlit as st
from pyomo.environ import *
import pandas as pd

# --------------------------
# Parameters and Sets
# --------------------------
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
shifts = ["day", "night"]

# Streamlit Sidebar Inputs
st.sidebar.title("Settings")
num_workers = st.sidebar.slider("Number of workers", min_value=2, max_value=20, value=10)
worker_names = st.sidebar.text_input(
    "Worker Names (comma separated)", 
    value=", ".join([f"W{i}" for i in range(1, num_workers + 1)])
)
num_stations = st.sidebar.slider("Number of stations", min_value=1, max_value=10, value=4)

# Parse workers
workers = [name.strip() for name in worker_names.split(",") if name.strip()]
stations = list(range(num_stations+1))

# --------------------------
# Helper Functions
# --------------------------
def build_model(workers, stations):
    model = ConcreteModel()

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

    model.needed = Var(workers, within=Binary, initialize=0)
    model.total_hours = Var(workers, within=NonNegativeReals, initialize=0)
    model.deviation = Var(workers, within=NonNegativeReals)

    average_hours = len(stations) * len(days) * len(shifts) / len(workers)

    model.obj = Objective(
        expr=(
            sum(
                model.works[worker, station, day, shift]
                for worker in workers
                for station in stations
                for day in days
                for shift in shifts
            )
            -  sum(model.deviation[worker] for worker in workers)
        ),
        sense=maximize,
    )

    model.constraints = ConstraintList()

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

    for day in days:
        for shift in shifts:
            for worker in workers:
                model.constraints.add(
                    sum(model.works[worker, station, day, shift] for station in stations)
                    <= 1
                )

    for station in stations:
        for day in days:
            for worker in workers:
                model.constraints.add(
                    model.works[worker, station, day, "day"]
                    == model.works[worker, station, day, "night"]
                )

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

    for station in stations:
        for day in days:
            for shift in shifts:
                model.constraints.add(
                    sum(model.works[worker, station, day, shift] for worker in workers)
                    <= 2
                )

    for worker in workers:
        model.constraints.add(
            model.deviation[worker] >= model.total_hours[worker] - average_hours
        )
        model.constraints.add(
            model.deviation[worker] >= average_hours - model.total_hours[worker]
        )

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

    return model

def solve_model(model):
    opt = SolverFactory("cbc")
    opt.options["sec"] = 10
    results = opt.solve(model, tee=True)
    return results

def get_workers_needed(model, workers):
    return [worker for worker in workers if model.needed[worker].value == 1]

def get_work_table(model, workers, stations):
    schedule = {
        station: {day: {shift: [] for shift in shifts} for day in days}
        for station in stations
    }
    for worker in workers:
        for station in stations:
            for day in days:
                for shift in shifts:
                    if model.works[worker, station, day, shift].value == 1:
                        schedule[station][day][shift].append(worker)
    return schedule

def generate_worker_schedule(model, workers, stations):
    worker_schedule = {}
    for worker in workers:
        worker_schedule[worker] = []
        for station in stations:
            for day in days:
                for shift in shifts:
                    if model.works[worker, station, day, shift].value == 1:
                        worker_schedule[worker].append(f"Station {station} - {day} {shift}")
    return worker_schedule

# --------------------------
# Streamlit App
# --------------------------
st.title("Worker Shift Scheduling Optimization")

if st.button("Run Optimization"):
    with st.spinner("Solving optimization model..."):
        model = build_model(workers, stations)
        solve_model(model)

        workers_needed = get_workers_needed(model, workers)
        week_table = get_work_table(model, workers, stations)
        worker_schedule = generate_worker_schedule(model, workers, stations)

    st.success("Optimization complete!")

    st.subheader("Workers Needed")
    st.write(workers_needed)

    st.subheader("Shift Schedule per Station")
    for station in stations:
        st.markdown(f"### Station {station}")
        table_data = []
        for day in days:
            row = {"Day": day}
            for shift in shifts:
                assigned = ", ".join(week_table[station][day][shift])
                row[shift] = assigned
            table_data.append(row)
        df = pd.DataFrame(table_data)
        st.dataframe(df)

    st.subheader("Worker-based Schedule")
    for worker, assignments in worker_schedule.items():
        st.markdown(f"**{worker}**")
        if assignments:
            for assignment in assignments:
                st.write(f"- {assignment}")
        else:
            st.write("- (No shifts assigned)")

st.sidebar.info(
    "Select the number of workers, input worker names (comma separated), and number of stations."
    " Then press 'Run Optimization'!"
)
