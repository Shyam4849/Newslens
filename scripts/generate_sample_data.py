"""Generates a realistic synthetic news dataset (data/news.csv) so the app runs out-of-the-box.

Replace with a real dataset (e.g. Kaggle 'News Category Dataset' or 'All the News')
by uploading a CSV/JSON in the app sidebar.  Columns: title, description, source, url, published, author, category
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

random.seed(7)
SOURCES = ["BBC", "Reuters", "The Guardian", "CNN", "NDTV", "Times of India", "TechCrunch",
           "Al Jazeera", "The Hindu", "Bloomberg", "Financial Times", "ESPNcricinfo"]
DAYS = 14
END = datetime(2026, 9, 29)

TOPICS = {
 "Technology": dict(
  name="ai_agents", weights=[1, 1, 2, 2, 3, 4, 5, 5, 6, 8, 10, 14, 18, 24],
  sources=["TechCrunch", "Reuters", "The Guardian", "BBC", "Bloomberg", "CNN", "Financial Times"],
  slots=dict(co=["OpenAI", "Google", "Microsoft", "Anthropic", "Meta", "Amazon", "Salesforce", "Nvidia"],
             what=["AI agent", "agentic AI platform", "autonomous coding agent", "AI assistant", "agent framework"],
             who=["developers", "enterprises", "customer support teams", "software engineers", "businesses"]),
  titles=["{co} launches new {what} for {who}", "{co} unveils autonomous {what} to automate workflows",
          "{co} expands {what} offering as AI automation race heats up", "Inside {co}'s push into agentic AI and automation",
          "{co} adds multi-step reasoning to its {what}", "Startups race to build {what} tools after {co} announcement"],
  descs=["The new agents can plan, call tools and complete multi-step tasks with little human supervision.",
         "Analysts say AI agents and workflow automation are becoming the next big battleground for large language models.",
         "The company said its generative AI models can now operate software, write code and handle support tickets autonomously.",
         "Developers get an SDK to build agents that use tools, memory and APIs, the firm said.",
         "Enterprise adoption of AI agents is accelerating as firms look to automate repetitive knowledge work."]),
 "Technology ": dict(
  name="chips", weights=[6, 6, 7, 6, 7, 8, 8, 9, 9, 10, 11, 11, 12, 13],
  sources=["Reuters", "Bloomberg", "Financial Times", "TechCrunch", "Times of India", "CNN"],
  slots=dict(co=["Nvidia", "AMD", "Intel", "TSMC", "Samsung", "Qualcomm", "Micron"],
             what=["AI accelerator", "GPU architecture", "3nm chip", "memory chip", "data-centre processor"],
             where=["Arizona", "Taiwan", "Gujarat", "Texas", "South Korea"]),
  titles=["{co} announces new {what} to meet AI demand", "{co} expands semiconductor fab in {where}",
          "Chip shortage fears return as {co} warns on {what} supply", "{co} unveils next-generation {what}",
          "Semiconductor stocks rally after {co} earnings beat", "US export curbs on chips hit {co} outlook"],
  descs=["Demand for advanced semiconductors used in AI training is outpacing foundry capacity.",
         "The chipmaker said its new silicon delivers major gains in performance per watt for data centres.",
         "Wafer fabrication capacity and packaging remain the key bottleneck for the global chip industry.",
         "Investors are watching semiconductor supply chains as export controls tighten.",
         "The fab is expected to begin volume production of advanced-node chips next year."]),
 "Business": dict(
  name="india_econ", weights=[9, 8, 9, 9, 8, 9, 9, 8, 9, 9, 8, 9, 9, 8],
  sources=["Times of India", "NDTV", "The Hindu", "Bloomberg", "Reuters", "Financial Times"],
  slots=dict(metric=["inflation", "GDP growth", "repo rate", "rupee", "industrial output", "retail inflation"],
             dir=["eases", "rises", "steady", "hits record", "slows"],
             who=["RBI", "the finance ministry", "Sensex", "Nifty", "economists"]),
  titles=["India's {metric} {dir} as {who} watch closely", "RBI holds {metric} decision; markets react",
          "{who} expect {metric} to stay steady in coming quarter", "Indian economy: {metric} {dir} in latest data",
          "Sensex and Nifty move as {metric} data surprises", "Rupee, {metric} and Indian markets: what to expect"],
  descs=["The Reserve Bank of India kept its monetary policy stance unchanged citing inflation and growth trends.",
         "Economists said India's GDP momentum remains strong even as global headwinds persist.",
         "Indian equity markets reacted to the latest macro data on inflation and industrial production.",
         "The finance ministry said fiscal deficit targets remain on track for the year.",
         "Foreign portfolio flows and the rupee were also in focus for Dalal Street investors."]),
 "Environment": dict(
  name="climate", weights=[3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9],
  sources=["The Guardian", "BBC", "Al Jazeera", "Reuters", "The Hindu", "CNN"],
  slots=dict(who=["EU", "India", "China", "United States", "UN", "G20"],
             what=["carbon tax", "emissions target", "renewable energy subsidy", "climate pact", "coal phase-out"],
             ev=["COP31", "climate summit", "climate talks"]),
  titles=["{who} agrees new {what} ahead of {ev}", "{ev}: countries clash over {what}",
          "Climate policy: {who} tightens {what}", "{who} pledges deeper {what} cuts amid heatwave",
          "Activists slam {who} over weak {what}", "Renewable push: {who} boosts {what}"],
  descs=["Negotiators said the deal aims to cut greenhouse gas emissions and accelerate the shift to clean energy.",
         "Scientists warned that current climate commitments still fall short of the 1.5C goal.",
         "The policy would price carbon emissions and expand solar and wind capacity.",
         "Environmental groups called the pledge a step forward but said enforcement remains unclear.",
         "Extreme weather events have added urgency to global climate negotiations."]),
 "Sports": dict(
  name="cricket", weights=[10, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3],
  sources=["ESPNcricinfo", "NDTV", "Times of India", "The Hindu", "BBC", "The Guardian"],
  slots=dict(team=["India", "Australia", "England", "Pakistan", "South Africa", "New Zealand"],
             player=["Kohli", "Bumrah", "Root", "Smith", "Rohit", "Starc"],
             what=["Test", "ODI", "T20I", "IPL match", "series decider"]),
  titles=["{player} stars as {team} win {what}", "{team} beat rivals in thrilling {what}",
          "{player} century powers {team} in {what}", "Cricket: {team} name squad for upcoming {what}",
          "{player} ruled out of {what} with injury scare", "Bowlers dominate as {team} clinch {what}"],
  descs=["A brilliant innings and disciplined bowling sealed the result on the final day of the match.",
         "The captain praised the batting unit after a superb team performance at the ground.",
         "Wickets tumbled in the second session before the middle order rebuilt the innings.",
         "Selectors announced the squad, with the wicketkeeper and spinners retaining their places.",
         "Fans packed the stadium as the run chase went down to the last over."]),
 "Science": dict(
  name="space", weights=[4, 4, 5, 4, 5, 5, 4, 5, 5, 4, 5, 5, 4, 5],
  sources=["BBC", "Reuters", "The Guardian", "The Hindu", "CNN", "TechCrunch"],
  slots=dict(agency=["NASA", "ISRO", "ESA", "SpaceX", "Blue Origin", "CNSA"],
             mission=["lunar lander", "Mars rover", "crewed capsule", "space telescope", "Gaganyaan mission"]),
  titles=["{agency} launches {mission} in major milestone", "{agency} delays {mission} after technical checks",
          "{agency} successfully tests {mission} ahead of launch", "Space race: {agency} reveals {mission} plans",
          "{agency} rocket lifts off carrying {mission}"],
  descs=["The rocket lifted off from the launch pad and deployed its payload into orbit as planned.",
         "Engineers completed final checks on the spacecraft ahead of the scheduled launch window.",
         "The mission aims to study the lunar surface and prepare for future crewed exploration.",
         "Astronomers hope the observatory will capture unprecedented images of distant galaxies."]),
 "Health": dict(
  name="health", weights=[5, 5, 4, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5],
  sources=["BBC", "The Guardian", "The Hindu", "Al Jazeera", "Times of India", "Reuters"],
  slots=dict(what=["vaccine", "dengue outbreak", "cancer drug", "flu season", "obesity drug", "antibiotic resistance"],
             who=["WHO", "ICMR", "FDA", "researchers", "health ministry"]),
  titles=["{who} approves new {what} after trials", "{who} warns of rising {what} cases",
          "Study: {what} shows promise, say {who}", "{who} issues advisory as {what} spreads",
          "Hospitals brace for {what} surge"],
  descs=["Clinical trial data showed a significant reduction in hospitalisations among patients.",
         "Health officials urged the public to seek early treatment and follow vaccination guidance.",
         "The study of thousands of patients was published in a peer-reviewed medical journal.",
         "Hospitals reported an increase in admissions and stressed the need for preventive care."]),
 "Politics": dict(
  name="politics", weights=[6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
  sources=["Al Jazeera", "BBC", "CNN", "Reuters", "The Guardian", "NDTV"],
  slots=dict(who=["Prime Minister", "President", "opposition leader", "coalition", "parliament"],
             what=["election", "trade deal", "sanctions", "peace talks", "budget bill", "no-confidence vote"],
             where=["Brussels", "Washington", "New Delhi", "Kyiv", "London"]),
  titles=["{who} faces {what} pressure in {where}", "Leaders meet in {where} for {what}",
          "{who} defends {what} as talks resume in {where}", "Opposition slams {who} over {what}",
          "{what}: what {who} said in {where} today"],
  descs=["Lawmakers debated the proposal for hours before the vote was postponed.",
         "Diplomats said talks were constructive but significant differences remain between the two sides.",
         "Opinion polls suggest the race remains tight ahead of the vote.",
         "The government said it would present its position to parliament next week."]),
}


def fill(t, slots):
    out = t
    for k, v in slots.items():
        out = out.replace("{" + k + "}", random.choice(v))
    return out


rows = []
for cat, spec in TOPICS.items():
    cat = cat.strip()
    for d, n in enumerate(spec["weights"]):
        day = END - timedelta(days=DAYS - 1 - d)
        stories = max(1, round(n * 0.55))
        for _ in range(stories):
            title = fill(random.choice(spec["titles"]), spec["slots"])
            desc = " ".join(random.sample(spec["descs"], k=min(2, len(spec["descs"]))))
            # the same story is carried by several outlets (-> duplicate detection demo)
            n_outlets = random.choice([1, 1, 2, 3, 4])
            for src in random.sample(spec["sources"], k=min(n_outlets, len(spec["sources"]))):
                t = title if random.random() < 0.5 else title.replace("launches", "unveils").replace("announces", "reveals")
                rows.append(dict(title=t, description=desc, source=src,
                                 url=f"https://example.com/{spec['name']}/{len(rows)}",
                                 published=(day + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))).isoformat(),
                                 author="", category=cat))
random.shuffle(rows)
df = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True)
df.to_csv("data/news.csv", index=False)
print(f"wrote data/news.csv with {len(df)} articles")
