from enum import Enum
import random
import statistics

class States(Enum):
  BAR = 1
  BELL = 2
  LEMON = 3
  CHERRY = 4

def payout(result, balance):
  """Takes in a slot-machine result and a balance and 
  updates the balance accordingly to a fixed payout scheme """

  # Checks for 1, 2 or 3 CHERRIES
  if result[0] == States.CHERRY.value:
    if result[1] == States.CHERRY.value:
      if result[2] == States.CHERRY.value:
        balance += 3
      else:
        balance += 2
    else:
      balance += 1

  # Check for 3 LEMONS
  if result[0] == States.LEMON.value and result[0] == result[1] == result[2]:
    balance += 5

  # Check for 3 BELLS
  if result[0] == States.BELL.value and result[0] == result[1] == result[2]:
    balance += 15
  
  # If all wheels are BAR
  if result[0] == States.BAR.value and result[0] == result[1] == result[2]:
    balance += 20
  
  # Does a coin bet
  balance -= 1

  return balance

def slot_Simulation(games):
  """Takes in the total number of games to be simulated"""

  # Empty list will eventually hold number of plays for each "game"
  plays = []

  # Loops over 100 000 "games" to collect a dataset with number of plays per game before the player goes broke
  for n in range(games):

    # Number of coins by default per "game" and plays per game
    init_Balance = 10
    game_Plays = 0

    # Play as long as not broke
    while (init_Balance > 0):

      # Does a "play"
      game_Plays += 1

      # Picks three random values for each wheel
      x = random.randint(1,4)
      y = random.randint(1,4)
      z = random.randint(1,4)

      # Adds the result to a array 
      iteration_Result = [x, y, z]

      init_Balance = payout(iteration_Result, init_Balance)

    # Stores the number of plays per "game"
    plays.append(game_Plays)

  play_Mean = statistics.mean(plays)
  play_Median = statistics.median(plays)

  print("The average number of plays for this slot machine one can expect to make before going broke is " + str(play_Mean) + " plays.")
  print("The median of plays is " + str(play_Median) + " plays.")
  
  return None

def generate_Birthdays(birthday_amount):

  # Empty list to hold all generated birthdays
  birthdays = []
  
  # Generate a fixed amount of birthdays
  for i in range(birthday_amount):

    # Picks a random integer between 1 and 365
    birthday = random.randint(1, 365)

    # Adds the generated date to the list of birthdays
    birthdays.append(birthday)

  return birthdays

def check_Occurence(birthdays):
  
  # False by default
  occurrence = False
  
  # Make a set of unique birthdays
  unique_birthdays = set(birthdays)
  
  # Count total amount of birthdays (SHOULD BE EQUAL TO SIZE)
  total_birthdays = len(birthdays)

  # Count total amount of unique birthdays
  total_unique_birthdays = len(unique_birthdays)

  # If number of unique birthdays and total birthdays not are equal, there must be duplicates and thus an occurrence of same the birth date
  occurrence = total_birthdays != total_unique_birthdays

  return occurrence

def birthday_Simulation(people_size, simulation_size):

  # 0 by default
  occurrences = 0

  # How many simulations one would like to perform
  for i in range(simulation_size):

    # Generates a list of birthdays as well as checks for occurrence in the list
    birthdays = generate_Birthdays(people_size)
    occurence = check_Occurence(birthdays) 

    # If there is an occurrence the count of occurrences is incremented by one 
    if occurence:
      occurrences += 1

  # Calculates the probablity of an occurrence by evidence/total number of trials
  probability = occurrences/simulation_size
  
  return probability

def birthday_range(min_people, max_people, simulation_size):
  
  # Empty list will hold probabilities for each N of people
  probabilities = []

  # Iterate over all possible N's of people
  for people_size in range(min_people, max_people + 1):

    # Calculate probability for each size of N people
    probability = birthday_Simulation(people_size, simulation_size)

    # Add the probability to the list of probabilities
    probabilities.append(probability)
  
  return probabilities

def all_days(simulation_size):

  # Empty list to hold all possible number of people
  number_Of_Peoples = []

  # How many simulations one would like to perform
  for i in range(simulation_size):

    # Empty set to hold all unique birthdays
    group = set()

    # 0 by default. Will be used to count how many people one should 
    # expect to form in order to meet the criteria of each day being 
    # a birthday
    count = 0

    # 2. Check whether all days of the year are covered.
    while (len(group) < 365):

      # Increment the number of people to include
      count += 1

      # Picks a random birthday
      random_person = random.randint(1, 365)

      # 1. Add a random person to the group
      group.add(random_person)

      # 2. Check whether all days of the year are covered.
    
    # Add how many people were needed to include for this trial 
    # to the list of numbers of people for all the trials

    number_Of_Peoples.append(count)
  
  # Calculate the average number of people to include in a group 
  # such that each day in a year is a birthday
  mean_Of_People = statistics.mean(number_Of_Peoples)

  return mean_Of_People

if __name__ == '__main__':
  #slot_Simulation(100000)
  simulation_size = 10000
  people_size = 23
  #count = all_days(simulation_size)
  #print(f"One must expect to include {count} people in a group before each day in a year is a birthday for the group.")
  #probabilities = birthday_range(10, 50, simulation_size)
  #for var in probabilities:
  #print(f"The probability is {var} for a occurrence of same birth date among {probabilities.index(var) + 10} people.")