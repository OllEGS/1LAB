"""Дискретно-событийная модель одного заказа в распределённой доставке."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from heapq import heappop, heappush


ROADS = {
    ("W", "H"): 4, ("H", "D"): 5, ("W", "D"): 12,
    ("B", "W"): 6, ("B", "H"): 3, ("W", "J"): 6, ("J", "D"): 7,
}


def shortest_path(start, goal, closed=frozenset()):
    """Алгоритм Дейкстры для двусторонней сети с положительными весами."""
    frontier = [(0, (start,))]
    settled = set()
    while frontier:
        minutes, path = heappop(frontier)
        vertex = path[-1]
        if vertex in settled:
            continue
        if vertex == goal:
            return minutes, path
        settled.add(vertex)
        for (left, right), cost in ROADS.items():
            if frozenset((left, right)) in closed:
                continue
            neighbor = {left: right, right: left}.get(vertex)
            if neighbor is not None and neighbor not in settled:
                heappush(frontier, (minutes + cost, path + (neighbor,)))
    return None


@dataclass
class Order:
    id: str = "O1"
    destination: str = "D"
    kilograms: int = 2
    deadline: int = 20
    status: str = "NEW"
    epoch: int = 0
    assigned: str | None = None


@dataclass
class Agent:
    name: str


@dataclass
class Client(Agent):
    status: str = "NEW"

    def submit(self, model):
        model.send(self.name, "DSP", "REQUEST", f"{model.order.id} → D, 2 кг, срок 20 мин")

    def receive(self, model, status):
        self.status = status
        model.send("DSP", self.name, "INFORM", status)


@dataclass
class Courier(Agent):
    position: str
    capacity: int
    mode: str = "GROUND"
    limits: str = "ROAD"
    state: str = "IDLE"

    def quote(self, model, pickup):
        if self.state != "IDLE" or self.capacity < model.order.kilograms:
            return None
        if self.mode == "AIR":
            # В учебной карте воздушные коридоры отсутствуют.
            return None
        first = shortest_path(self.position, pickup, model.closed)
        second = shortest_path(pickup, model.order.destination, model.closed)
        if first is None or second is None:
            return None
        return first[0] + second[0]

    def move(self, model, goal):
        route = shortest_path(self.position, goal, model.closed)
        if route is None:
            return False
        minutes, path = route
        if minutes:
            model.time += minutes
            model.event(f"{self.name}: {' → '.join(path)} ({minutes} мин)")
        self.position = goal
        return True


@dataclass
class Facility(Agent):
    position: str
    slots: int = 1
    parcel_epoch: int | None = None

    def release(self, model, courier, epoch):
        if (self.parcel_epoch is None or self.parcel_epoch > epoch
                or not model.dispatcher.active(model.order, courier, epoch)
                or courier.position != self.position or model.holder != self.name):
            return False
        self.parcel_epoch = None
        model.holder = courier.name
        model.dispatcher.confirmed_holder = courier.name
        model.send(self.name, "DSP", "PICKED_UP", f"{courier.name}, v{epoch}")
        return True


@dataclass
class TransportHub(Facility):
    def receive(self, model, courier, epoch):
        if (self.slots < 1 or self.parcel_epoch is not None
                or not model.dispatcher.active(model.order, courier, epoch)
                or courier.position != self.position or model.holder != courier.name):
            return False
        self.parcel_epoch = epoch
        model.holder = self.name
        model.dispatcher.confirmed_holder = self.name
        model.send(self.name, "DSP", "AT_HUB", f"{self.name}, v{epoch}")
        return True


@dataclass
class Dispatcher(Agent):
    confirmed_holder: str = "W"

    @staticmethod
    def active(order, courier, epoch):
        return order.assigned == courier.name and order.epoch == epoch

    def assign(self, model, facility):
        order = model.order
        if self.confirmed_holder != facility.name or order.assigned is not None or order.status != "QUEUED":
            return None
        eligible_quotes = []
        for courier in model.couriers:
            quote = courier.quote(model, facility.position)
            if quote is not None:
                eligible_quotes.append((quote, courier.name, courier))
        if not eligible_quotes:
            order.status = "PENDING"
            model.client.receive(model, order.status)
            return None
        minutes, _, courier = min(eligible_quotes)
        order.epoch += 1
        order.assigned = courier.name
        order.status = "ASSIGNED"
        courier.state = "BUSY"
        model.send("DSP", courier.name, "ASSIGN", f"{order.id}, v{order.epoch}, ETA={model.time + minutes}")
        model.send(courier.name, "DSP", "ACCEPT", f"{order.id}, v{order.epoch}")
        return courier

    def fail(self, model, courier, *, reported):
        order = model.order
        model.time += 2
        if reported:
            model.send(courier.name, "DSP", "FAILURE", f"{order.id}, v{order.epoch}")
        else:
            model.event("Истёк тайм-аут ожидания связи с курьером")
        courier.state = "FAILED"
        order.assigned = None
        if self.confirmed_holder == courier.name:
            order.status = "RECOVERY"
            model.client.receive(model, order.status)
            return None
        order.status = "QUEUED"
        facility = model.facilities[self.confirmed_holder]
        return self.assign(model, facility)

    def deliver(self, model, courier, epoch):
        order = model.order
        if (order.status != "IN_TRANSIT" or not self.active(order, courier, epoch)
                or model.holder != courier.name or courier.position != order.destination):
            return False
        model.send(courier.name, "DSP", "DELIVERED", f"{order.id}, v{epoch}")
        order.status = "DELIVERED"
        order.assigned = None
        courier.state = "IDLE"
        model.holder = model.client.name
        self.confirmed_holder = model.client.name
        model.client.receive(model, order.status)
        return True


class Simulation:
    def __init__(self, scenario):
        self.scenario = scenario
        self.time = 0
        self.messages = 0
        self.order = Order()
        self.client = Client("C1")
        self.dispatcher = Dispatcher("DSP")
        self.couriers = [Courier("K1", "W", 3), Courier("K2", "B", 5)]
        self.facilities = {"W": Facility("W", "W", parcel_epoch=0),
                           "H": TransportHub("H", "H")}
        self.holder = "W"
        self.closed = set()

    def event(self, label):
        print(f"[{self.time:02} мин] {label}")

    def send(self, sender, target, verb, detail):
        self.messages += 1
        self.event(f"{sender} → {target} | {verb} | {detail}")

    def pick_up(self, courier):
        facility = self.facilities[self.holder]
        if not courier.move(self, facility.position):
            return False
        self.send(courier.name, facility.name, "PICKUP", f"{self.order.id}, v{self.order.epoch}")
        if not facility.release(self, courier, self.order.epoch):
            return False
        self.order.status = "IN_TRANSIT"
        return True

    def run(self):
        self.client.submit(self)
        self.send("DSP", "W", "CHECK_READY", self.order.id)
        self.send("W", "DSP", "READY", self.order.id)
        self.order.status = "QUEUED"
        courier = self.dispatcher.assign(self, self.facilities["W"])
        if courier is None:
            return
        if self.scenario == "fail-before":
            courier = self.dispatcher.fail(self, courier, reported=True)
            if courier is None:
                return
        if not self.pick_up(courier):
            raise RuntimeError("Передача посылки не подтверждена")
        if self.scenario in {"fail-hub", "lost-contact", "road-closed"}:
            if not courier.move(self, "H"):
                raise RuntimeError("Не удалось добраться до узла H")
            if self.scenario == "lost-contact":
                self.dispatcher.fail(self, courier, reported=False)
                return
            if self.scenario == "fail-hub":
                self.send(courier.name, "H", "HANDOFF", f"{self.order.id}, v{self.order.epoch}")
                if not self.facilities["H"].receive(self, courier, self.order.epoch):
                    raise RuntimeError("Узел не подтвердил приём")
                courier = self.dispatcher.fail(self, courier, reported=True)
                if courier is None:
                    return
                if not self.pick_up(courier):
                    raise RuntimeError("Резервный курьер не получил посылку")
            if self.scenario == "road-closed":
                self.closed.add(frozenset(("H", "D")))
                self.send("H", "DSP", "ROAD_BLOCKED", "H–D")
                self.send("DSP", courier.name, "MAP_UPDATE", "H–D закрыто")
        if not courier.move(self, self.order.destination):
            self.order.status = "RECOVERY"
            self.client.receive(self, self.order.status)
            return
        if not self.dispatcher.deliver(self, courier, self.order.epoch):
            raise RuntimeError("Доставка не подтверждена")


SCENARIOS = ("normal", "fail-before", "fail-hub", "lost-contact", "road-closed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=(*SCENARIOS, "all"), nargs="?", default="all")
    args = parser.parse_args()
    for name in SCENARIOS if args.scenario == "all" else (args.scenario,):
        print(f"\n=== {name} ===")
        model = Simulation(name)
        model.run()
        order = model.order
        on_time = order.status == "DELIVERED" and model.time <= order.deadline
        print(f"Итог: {order.status} | t={model.time} мин | сообщений={model.messages}"
              f" | версия={order.epoch} | держатель={model.holder} | в срок={'да' if on_time else 'нет'}")


if __name__ == "__main__":
    main()
