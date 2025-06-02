# ml_api/models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime

class TrainRequest(BaseModel):
    """
    Request body for the /train endpoint.
    In a real app, you might pass data directly or a data source identifier.
    For this example, we'll use a fixed dummy dataset.
    """
    model_type: str = "RandomForestClassifier"
    n_estimators: int = 100
    random_state: int = 42
    importance_threshold: float = 0.005 # Threshold for feature selection

class TrainResponse(BaseModel):
    """
    Response body for the /train endpoint.
    """
    message: str
    model_saved_path: str
    num_features_selected: int



class CustomerLoanRecord(BaseModel):
    """
    Pydantic model representing a single row of the customer loan dataset.

    Data types are inferred from the provided CSV header and example row.
    All numerical features are typed as Optional[float] to accommodate
    potential missing values (NaN) and decimal numbers.
    String fields are Optional[str] for similar reasons.
    Date fields are Optional[datetime].
    """
    customerid: str
    loanid: str
    date_add: datetime # Assuming this will be parsed into a datetime object
    prodcode: str

    # Features 1 to 64 - inferred as float due to potential decimals/NaNs
    feature_1: Optional[float] = Field(None, alias='feature_1')
    feature_2: Optional[float] = Field(None, alias='feature_2')
    feature_3: Optional[float] = Field(None, alias='feature_3')
    feature_4: Optional[float] = Field(None, alias='feature_4')
    feature_5: Optional[float] = Field(None, alias='feature_5')
    feature_6: Optional[float] = Field(None, alias='feature_6')
    feature_7: Optional[float] = Field(None, alias='feature_7')
    feature_8: Optional[float] = Field(None, alias='feature_8')
    feature_9: Optional[float] = Field(None, alias='feature_9')
    feature_10: Optional[float] = Field(None, alias='feature_10')
    feature_11: Optional[float] = Field(None, alias='feature_11')
    feature_12: Optional[float] = Field(None, alias='feature_12')
    feature_13: Optional[float] = Field(None, alias='feature_13')
    feature_14: Optional[float] = Field(None, alias='feature_14')
    feature_15: Optional[float] = Field(None, alias='feature_15')
    feature_16: Optional[float] = Field(None, alias='feature_16')
    feature_17: Optional[float] = Field(None, alias='feature_17')
    feature_18: Optional[float] = Field(None, alias='feature_18')
    feature_19: Optional[float] = Field(None, alias='feature_19')
    feature_20: Optional[float] = Field(None, alias='feature_20')
    feature_21: Optional[float] = Field(None, alias='feature_21')
    feature_22: Optional[float] = Field(None, alias='feature_22')
    feature_23: Optional[float] = Field(None, alias='feature_23')
    feature_24: Optional[float] = Field(None, alias='feature_24')
    feature_25: Optional[float] = Field(None, alias='feature_25')
    feature_26: Optional[float] = Field(None, alias='feature_26')
    feature_27: Optional[float] = Field(None, alias='feature_27')
    feature_28: Optional[float] = Field(None, alias='feature_28')
    feature_29: Optional[float] = Field(None, alias='feature_29')
    feature_30: Optional[float] = Field(None, alias='feature_30')
    feature_31: Optional[float] = Field(None, alias='feature_31')
    feature_32: Optional[float] = Field(None, alias='feature_32')
    feature_33: Optional[float] = Field(None, alias='feature_33')
    feature_34: Optional[float] = Field(None, alias='feature_34')
    feature_35: Optional[float] = Field(None, alias='feature_35')
    feature_36: Optional[float] = Field(None, alias='feature_36')
    feature_37: Optional[float] = Field(None, alias='feature_37')
    feature_38: Optional[float] = Field(None, alias='feature_38')
    feature_39: Optional[float] = Field(None, alias='feature_39')
    feature_40: Optional[float] = Field(None, alias='feature_40')
    feature_41: Optional[float] = Field(None, alias='feature_41')
    feature_42: Optional[float] = Field(None, alias='feature_42')
    feature_43: Optional[float] = Field(None, alias='feature_43')
    feature_44: Optional[float] = Field(None, alias='feature_44')
    feature_45: Optional[float] = Field(None, alias='feature_45')
    feature_46: Optional[float] = Field(None, alias='feature_46')
    feature_47: Optional[float] = Field(None, alias='feature_47')
    feature_48: Optional[float] = Field(None, alias='feature_48')
    feature_49: Optional[float] = Field(None, alias='feature_49')
    feature_50: Optional[float] = Field(None, alias='feature_50')
    feature_51: Optional[float] = Field(None, alias='feature_51')
    feature_52: Optional[float] = Field(None, alias='feature_52')
    feature_53: Optional[float] = Field(None, alias='feature_53')
    feature_54: Optional[float] = Field(None, alias='feature_54')
    feature_55: Optional[float] = Field(None, alias='feature_55')
    feature_56: Optional[float] = Field(None, alias='feature_56')
    feature_57: Optional[float] = Field(None, alias='feature_57')
    feature_58: Optional[float] = Field(None, alias='feature_58')
    feature_59: Optional[float] = Field(None, alias='feature_59')
    feature_60: Optional[float] = Field(None, alias='feature_60')
    feature_61: Optional[float] = Field(None, alias='feature_61')
    feature_62: Optional[float] = Field(None, alias='feature_62')
    feature_63: Optional[float] = Field(None, alias='feature_63')
    feature_64: Optional[float] = Field(None, alias='feature_64')

    # Target/Indicator variables - inferred as int (0 or 1)
    default_ind: int
    write_off_ind: int

    class Config:
        # This allows Pydantic to handle extra fields gracefully if they appear
        # in the input data but are not defined in the model.
        # Set to False if you want strict validation.
        extra = "ignore"
        # For parsing datetime strings
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PredictResponse(BaseModel):
    """
    Response body for the /predict endpoint.
    """
    customerid: str
    loanid: str
    probability_of_default: float

class ExplainResponse(BaseModel):
    """
    Response body for the /explain endpoint.
    """
    predictions: List[int]
    shap_values: List[List[float]]
    feature_names: List[str]
    expected_value: float # Expected value for the positive class