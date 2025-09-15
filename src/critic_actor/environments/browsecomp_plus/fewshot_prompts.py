"""
Helpers for setting up few-shot prompts
"""
from typing import Any
import json
import re

from .data import render_prompt


FEWSHOT_SAMPLES = {
    "webact_mc_simple6": '\nQuestion: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\n\nOnly search from the possible searches below:\n0. search(title="Los Estoraques Unique Natural Area")\n1. search(title="Colorado orogeny")\n2. search(title="Vikos–Aoös National Park")\n3. search(title="Nevadan orogeny")\n4. search(title="Swall Meadows, California")\n5. search(title="Sourdough Glacier")\n6. search(title="Algoman orogeny")\n7. search(title="High Plains (United States)")\n8. search(title="Sierra Nevada lower montane forest")\n9. search(title="East European Platform")\n\nAction 1: Search[Colorado orogeny]\nObservation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\nAction 2: Lookup[eastern sector]\nObservation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\nAction 3: Search[High Plains]\nObservation 3: High Plains refers to one of two distinct land regions:\nAction 4: Search[High Plains (United States)]\nObservation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\nAction 5: Finish[1,800 to 7,000 ft]\n\nQuestion: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?\n\nOnly search from the possible searches below:\n0. search(title="Lisa Simpson")\n1. search(title="Marge Simpson")\n2. search(title="Bart Simpson")\n3. search(title="Allie Goertz")\n4. search(title="Milhouse Van Houten")\n5. search(title="Los Angeles Reader")\n6. search(title="Homer Simpson")\n7. search(title="List of The Simpsons video games")\n8. search(title="The Simpsons: An Uncensored, Unauthorized History")\n9. search(title="List of The Simpsons guest stars")\n\nAction 1: Search[Milhouse]\nObservation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.\nAction 2: Lookup[named after]\nObservation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. \nAction 3: Finish[Richard Nixon]\n\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n\nOnly search from the possible searches below:\n0. search(title="Adam (musical)")\n1. search(title="Adam Clayton Powell (film)")\n2. search(title="Adam Clayton Powell Jr.")\n3. search(title="Adam Clayton Powell IV")\n4. search(title="Seventh Avenue (Manhattan)")\n5. search(title="Mother African Methodist Episcopal Zion Church")\n6. search(title="Abyssinian Baptist Church")\n7. search(title="Adam Clayton Powell Jr. State Office Building")\n8. search(title="The Saimaa Gesture")\n9. search(title="Adam Clayton Powell Sr.")\n\nAction 1: Search[Adam Clayton Powell]\nObservation 1: Could not find [Adam Clayton Powell]. Similar: [\'Adam Clayton Powell III\', \'Seventh Avenue (Manhattan)\', \'Adam Clayton Powell Jr. State Office Building\', \'Isabel Washington Powell\', \'Adam Powell\', \'Adam Clayton Powell (film)\', \'Giancarlo Esposito\'].\nAction 2: Search[Adam Clayton Powell (film)]\nObservation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\nAction 3: Finish[The Saimaa Gesture]\n\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\n\nOnly search from the possible searches below:\n0. search(title="Baby Doll")\n1. search(title="Nicholas Ray")\n2. search(title="Elia Kazan")\n3. search(title="Art &amp; Gadg")\n4. search(title="A Letter to Elia")\n5. search(title="The Visitors (1972 film)")\n6. search(title="Truckline Cafe")\n7. search(title="All My Sons")\n8. search(title="Maya Kazan")\n9. search(title="Macao (film)")\n\nAction 1: Search[Nicholas Ray]\nObservation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\nAction 2: Search[Elia Kazan]\nObservation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\nAction 3: Finish[director, screenwriter, actor]\n\nQuestion: Which magazine was started first Arthur\'s Magazine or First for Women?\n\nOnly search from the possible searches below:\n0. search(title="Radio City (Indian radio station)")\n1. search(title="History of Albanian football")\n2. search(title="Echosmith")\n3. search(title="Women\'s colleges in the Southern United States")\n4. search(title="First Arthur County Courthouse and Jail")\n5. search(title="Arthur\'s Magazine")\n6. search(title="2014–15 Ukrainian Hockey Championship")\n7. search(title="First for Women")\n8. search(title="Freeway Complex Fire")\n9. search(title="William Rast")\n\nAction 1: Search[Arthur\'s Magazine]\nObservation 1: Arthur\'s Magazine (1844-\x80\x931846) was an American literary periodical published in Philadelphia in the 19th century. \nAction 2: Search[First for Women]\nObservation 2: First for Women is a woman\'s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. \nAction 3: Finish[Arthur\'s Magazine]\n\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n\nOnly search from the possible searches below:\n0. search(title="Leonid Ramzin")\n1. search(title="Leftover hash lemma")\n2. search(title="Phthinosaurus")\n3. search(title="Kate Dillon Levin")\n4. search(title="Tony Levin")\n5. search(title="Gabriel Levin")\n6. search(title="Leonid Levin")\n7. search(title="Pavel Urysohn")\n8. search(title="Meyer Levin")\n9. search(title="Leonid Bunimovich")\n\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (February 3, 1898 â\x80\x93 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \nAction 3: Finish[yes]\n',
    "webthink_mc_simple6": '\nQuestion: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\n\nOnly search from the possible searches below:\n0. search(title="Los Estoraques Unique Natural Area")\n1. search(title="Colorado orogeny")\n2. search(title="Vikos–Aoös National Park")\n3. search(title="Nevadan orogeny")\n4. search(title="Swall Meadows, California")\n5. search(title="Sourdough Glacier")\n6. search(title="Algoman orogeny")\n7. search(title="High Plains (United States)")\n8. search(title="Sierra Nevada lower montane forest")\n9. search(title="East European Platform")\n\nThought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.\nAction 1: Search[Colorado orogeny]\nObservation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\nThought 2: It does not mention the eastern sector. So I need to look up eastern sector.\nAction 2: Lookup[eastern sector]\nObservation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\nThought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.\nAction 3: Search[High Plains]\nObservation 3: High Plains refers to one of two distinct land regions:\nThought 4: I need to instead search High Plains (United States).\nAction 4: Search[High Plains (United States)]\nObservation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\nThought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\nAction 5: Finish[1,800 to 7,000 ft]\n\nQuestion: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?\n\nOnly search from the possible searches below:\n0. search(title="Lisa Simpson")\n1. search(title="Marge Simpson")\n2. search(title="Bart Simpson")\n3. search(title="Allie Goertz")\n4. search(title="Milhouse Van Houten")\n5. search(title="Los Angeles Reader")\n6. search(title="Homer Simpson")\n7. search(title="List of The Simpsons video games")\n8. search(title="The Simpsons: An Uncensored, Unauthorized History")\n9. search(title="List of The Simpsons guest stars")\n\nThought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.\nAction 1: Search[Milhouse]\nObservation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.\nThought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".\nAction 2: Lookup[named after]\nObservation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. \nThought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\nAction 3: Finish[Richard Nixon]\n\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n\nOnly search from the possible searches below:\n0. search(title="Adam (musical)")\n1. search(title="Adam Clayton Powell (film)")\n2. search(title="Adam Clayton Powell Jr.")\n3. search(title="Adam Clayton Powell IV")\n4. search(title="Seventh Avenue (Manhattan)")\n5. search(title="Mother African Methodist Episcopal Zion Church")\n6. search(title="Abyssinian Baptist Church")\n7. search(title="Adam Clayton Powell Jr. State Office Building")\n8. search(title="The Saimaa Gesture")\n9. search(title="Adam Clayton Powell Sr.")\n\nThought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\nAction 1: Search[Adam Clayton Powell]\nObservation 1: Could not find [Adam Clayton Powell]. Similar: [\'Adam Clayton Powell III\', \'Seventh Avenue (Manhattan)\', \'Adam Clayton Powell Jr. State Office Building\', \'Isabel Washington Powell\', \'Adam Powell\', \'Adam Clayton Powell (film)\', \'Giancarlo Esposito\'].\nThought 2: To find the documentary, I can search Adam Clayton Powell (film).\nAction 2: Search[Adam Clayton Powell (film)]\nObservation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\nThought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\nAction 3: Finish[The Saimaa Gesture]\n\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\n\nOnly search from the possible searches below:\n0. search(title="Baby Doll")\n1. search(title="Nicholas Ray")\n2. search(title="Elia Kazan")\n3. search(title="Art &amp; Gadg")\n4. search(title="A Letter to Elia")\n5. search(title="The Visitors (1972 film)")\n6. search(title="Truckline Cafe")\n7. search(title="All My Sons")\n8. search(title="Maya Kazan")\n9. search(title="Macao (film)")\n\nThought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\nAction 1: Search[Nicholas Ray]\nObservation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\nThought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\nAction 2: Search[Elia Kazan]\nObservation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\nThought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\nAction 3: Finish[director, screenwriter, actor]\n\nQuestion: Which magazine was started first Arthur\'s Magazine or First for Women?\n\nOnly search from the possible searches below:\n0. search(title="Radio City (Indian radio station)")\n1. search(title="History of Albanian football")\n2. search(title="Echosmith")\n3. search(title="Women\'s colleges in the Southern United States")\n4. search(title="First Arthur County Courthouse and Jail")\n5. search(title="Arthur\'s Magazine")\n6. search(title="2014–15 Ukrainian Hockey Championship")\n7. search(title="First for Women")\n8. search(title="Freeway Complex Fire")\n9. search(title="William Rast")\n\nThought 1: I need to search Arthur\'s Magazine and First for Women, and find which was started first.\nAction 1: Search[Arthur\'s Magazine]\nObservation 1: Arthur\'s Magazine (1844-\x80\x931846) was an American literary periodical published in Philadelphia in the 19th century. \nThought 2: Arthur\'s Magazine was started in 1844. I need to search First for Women next.\nAction 2: Search[First for Women]\nObservation 2: First for Women is a woman\'s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. \nThought 3: First for Women was started in 1989. 1844 (Arthur\'s Magazine) < 1989 (First for Women), so Arthur\'s Magazine was started first.\nAction 3: Finish[Arthur\'s Magazine]\n\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n\nOnly search from the possible searches below:\n0. search(title="Leonid Ramzin")\n1. search(title="Leftover hash lemma")\n2. search(title="Phthinosaurus")\n3. search(title="Kate Dillon Levin")\n4. search(title="Tony Levin")\n5. search(title="Gabriel Levin")\n6. search(title="Leonid Levin")\n7. search(title="Pavel Urysohn")\n8. search(title="Meyer Levin")\n9. search(title="Leonid Bunimovich")\n\nThought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (February 3, 1898 â\x80\x93 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\nThought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \nThought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. \nAction 3: Finish[yes]\n',
}


def parse_action(string):
    """
    Parse action from "action_type[argument]" format
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)
    if match:
        action_type = match.group(1).lower()
        argument = match.group(2)
        return action_type, argument
    else:
        return None, None


def get_react_messages(
    system_prompt: str | None,
    num_samples: int = 1,
    prompt_type: str = "webthink_mc_simple6",
    tool_role: str = "tool",
    thought_tokens: bool = False,
    func_call_argname: str = "arguments",  # or "parameters"
    include_titles_in_prompt: bool = True,
    prior_obs_not_in_state: bool = False,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Get ReAct prompt with optional few-shot examples
    """
    count = 0
    samples = FEWSHOT_SAMPLES[prompt_type]
    # Format examples from Reflexion
    samples = (
        samples.replace("\nQuestion", "\n\n\nQuestion")
        .replace("\nThought", "\n\n\nThought")
        .replace("\nAction", "\n\n\nAction")
        .replace("\nObservation", "\n\n\nObservation")
    )
    samples = samples.split("\n\n\nQuestion: ")[1:]

    all_messages = []
    all_search_tool_desc = []
    for _sample_idx, sample in enumerate(samples):
        messages = [
            {"role": "system", "content": system_prompt}
        ] if system_prompt is not None else []

        if "Lookup" not in sample:  # Only do search and final answer functions
            for action in sample.split("\n\n\n"):
                if "Thought" in action:
                    content = action.split("Thought")[1][4:]
                    if thought_tokens:
                        content = f"<think>\n{content}\n</think>"

                elif "Action" in action:
                    # Tool call or finish answer
                    act_content = action.split("Action")[1][4:]
                    fc_name, fc_argval = parse_action(act_content)

                    if fc_name == "search":
                        func_call = {
                            "name": fc_name,
                            func_call_argname: {"title": fc_argval}
                        }
                        func_call = json.dumps(func_call)
                        act_content = f"<tool_call>\n{func_call}\n</tool_call>"

                    elif fc_name == "finish":
                        act_content = f"Final Answer: {fc_argval}"

                    try:
                        act_content = content + "\n\n" + act_content
                    except UnboundLocalError as e:
                        # We don't have "Thought" in the examples yet
                        print(e)
                        
                    messages.append({"role": "assistant", "content": act_content})
                    if f"Final Answer: {fc_argval}" in act_content:
                        response = "Correct!"
                        messages.append({"role": tool_role, "content": response})

                elif "Observation" in action:
                    obs_content = action.split("Observation")[1][4:]
                    obs_content = f"## Result:\n{obs_content}"
                    messages.append({"role": tool_role, "content": obs_content})
                    # del obs_content

                else:  # Question
                    user_content = action
                    # print(user_content)
                    query = user_content.split("\n")[0]
                    all_titles = [f'{t.split('")')[0][1:]}' for t in user_content.split("search(title=")[1:]]
                    user_content, search_tool_desc = render_prompt(
                        query=query, all_titles=all_titles, include_titles_in_prompt=include_titles_in_prompt
                    )
                    messages.append(
                        {"role": "user", "content": user_content}
                    )
                    all_search_tool_desc.append(search_tool_desc)
                    count += 1

            if prior_obs_not_in_state:
                # Remove observations except for the last one
                _messages = [
                    m for m in messages[:-2] if m.get("role", None) != tool_role
                ]
                _messages.append(
                    {"role": tool_role, "content": obs_content}
                )
                messages = _messages + messages[-2:]

        
            all_messages.append(messages)
        
        if count >= num_samples:
            break
    return all_messages, all_search_tool_desc
