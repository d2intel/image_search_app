/**
 * Autogenerated by Thrift Compiler (0.11.0)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */

public enum SPRequestType implements org.apache.thrift.TEnum {
  IMAGE_SEARCH(0),
  SEARCH_DEBUG(1);

  private final int value;

  private SPRequestType(int value) {
    this.value = value;
  }

  /**
   * Get the integer value of this enum value, as defined in the Thrift IDL.
   */
  public int getValue() {
    return value;
  }

  /**
   * Find a the enum type by its integer value, as defined in the Thrift IDL.
   * @return null if the value is not found.
   */
  public static SPRequestType findByValue(int value) { 
    switch (value) {
      case 0:
        return IMAGE_SEARCH;
      case 1:
        return SEARCH_DEBUG;
      default:
        return null;
    }
  }
}
