# given an area name, predict its features.

# Example of dialog:
# ------------------
#
# Camden is good for nightlife and touristy
# nightlife is described by words like restaurant, dining, disco, midnight, saturday, night
# Crouch End and Finchley are not touristy
# touristy is described by words like tourists, museum, landscape, beautiful
# Islington is good for locals
# If an area is quiet, it is unlikely to be good for nightlife
# South Kensington is good for shopping and touristy
# Camdem is in the top 10 most populated area
# What are the areas good for nightlife?
# Can you describe Finchley?
# Is Islington touristy?
# What are the words that describe "good for shopping"?
#
# [((Area:Camden, good_for_nightlife), True), ((Area:Camden, touristy), True)]
# (Area:camden, Feature:top_10_population)



# from datetime import datetime



