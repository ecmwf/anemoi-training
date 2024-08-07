#########
 Testing
#########

It is strongly recommended that every time new features are added to the code, tests are written using the `pytest framework <https://docs.pytest.org/en/stable/>`_. 
These test should be designed so that if the feature is broken then the tests do not pass (and vice-versa)

This ensures that changes other developers make will not break features designed by other developers.