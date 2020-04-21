import sagemaker
# import Constants as cons


# def get_execution_role(sagemaker_session=None):
#     """
#     Returns the role ARN whose credentials are used to call the API.
#     In AWS notebook instance, this will return the ARN attributed to the
#     notebook. Otherwise, it will return the ARN stored in settings
#     at the project level.
#     :param: sagemaker_session(Session): Current sagemaker session
#     :rtype: string: the role ARN
#     """
#     try:
#         role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
#     except ValueError as e:
#         role = cons.ROLE
#     return role